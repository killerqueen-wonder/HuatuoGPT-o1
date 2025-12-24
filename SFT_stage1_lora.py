import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import wandb
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup
import shutil
import json
import traceback
from jinja2 import Template
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
os.umask(0)



logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        with open(config.data_path) as f:
            self.data = json.load(f)
        
        newdata = []
        for da in self.data:
            newdata.append(da)
        print('过滤掉',len(self.data),len(newdata))
        self.data = newdata

        self.max_seq_len = self.config.max_seq_len
        self.debug = 0

        # 如果从Base LLMs训练，选择 llama3-instruct作为模版
        chat_template_llama3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        if not tokenizer.chat_template:
            print('[debug]find no chat_templete')
            tokenizer.chat_template = chat_template_llama3
            
        self.template = Template(tokenizer.chat_template)

    def __getitem__(self, index):
        return self.data[index]

    def get_response(self,da):
        # temp = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        temp = '## Thinking\n\n{}\n\n## Final Response\n<answer>\n{}\n</answer>'
        return temp.format(da['Complex_CoT'],da['Response'])
        


    def get_prompt(self,da):

        # q = da['Question']
        q = da['Open-ended Verifiable Question']
        a = self.get_response(da)
        assert q is not None and a is not None, f'q:{q} a:{a}'

        # input =  self.template.render(messages=[{"role": "user", "content": q},{"role": "assistant", "content": a}],bos_token=self.tokenizer.bos_token,add_generation_prompt=False)
        # input_ids = self.tokenizer.encode(input,add_special_tokens= False)

        # query = self.template.render(messages=[{"role": "user", "content": q}],bos_token=self.tokenizer.bos_token,add_generation_prompt=True)
        # query_ids = self.tokenizer.encode(query,add_special_tokens= False)
        
        # Qwen3 直接使用 chat_template
        messages_full = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
        messages_query = [
            {"role": "user", "content": q}
        ]

        # 构造完整样本（user + assistant）
        input_ids = self.tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False
        )

        # 构造query部分（只保留user，用于mask标签）
        query_ids = self.tokenizer.apply_chat_template(
            messages_query,
            tokenize=True,
            add_generation_prompt=True
        )    
           
        labels = [-100]*len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)
        return {"input_ids": input_ids[-self.max_seq_len:], "labels": labels[-self.max_seq_len:]}        

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]

        max_len = max(len(x) for x in input_ids)
        print(f"[debug]max_len={max_len}===========================================")
        max_len = min(max_len,self.max_seq_len)
        print(f"[debug]max_len={max_len}===========================================")
        input_ids = [ item[:max_len] + [self.tokenizer.eos_token_id]*(max_len-len(item)) for item in input_ids]
        labels = [ item[:max_len] + [-100]*(max_len-len(item)) for item in labels]
        if self.debug < 3:
            print('input_ids',self.tokenizer.decode(input_ids[-1]))
            print('labels',self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            self.debug += 1

        return {
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
            }
    
    def __len__(self):
        return len(self.data)

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss


from peft import LoraConfig, get_peft_model, TaskType

def train(args):
    # 1. 初始化 Accelerator (确保 gradient_accumulation_steps 与 yaml 一致)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps) 

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, config=args, dir=args.log_dir, mode="online")
    
    accelerator.print(f'args:\n{args}')

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 3. 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        trust_remote_code=True
    )

    # 4. 注入 LoRA (解决 OOM 的核心)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # 5. 显存优化配置
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # 6. 优化器定义 (仅包含 LoRA 参数)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 7. 数据加载
    train_dataset = Train_dataset(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    # 8. 调度器设置
    num_training_steps = int(len(train_dataloader) * (args.n_epochs)) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_rates * num_training_steps), 
        num_training_steps=num_training_steps
    )
    
    # 9. 关键步：使用 accelerator.prepare 包装所有组件
    # DeepSpeed 引擎会在此步骤接管 optimizer 和 scheduler 的步进逻辑
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    metric = SFTMetric(device=torch.cuda.current_device())

    # 10. LoRA 专用保存逻辑
    def save_checkpoint(epoch, step, global_step):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, "tfmr")
            # 仅保存 LoRA 权重，极快且省空间
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, 
                save_function=accelerator.save,
                safe_serialization=True
            )
            tokenizer.save_pretrained(output_dir)
            
            # 拷贝必要的基础配置
            for item in ["config.json", "generation_config.json", "tokenizer_config.json"]:
                src = os.path.join(args.model_path, item)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(output_dir, item))

        accelerator.wait_for_everyone()
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))

    # 11. 训练循环 (移除了 accumulate 上下文)
    model.train()
    global_step = 0
    for epoch in range(args.n_epochs):
        train_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        
        for batch_cnt, batch in train_iter:
            input_ids = batch['input_ids']
            labels = batch['labels']

            # 前向传播
            output = model(input_ids=input_ids, labels=labels, return_dict=True, use_cache=False)
            loss = output.loss
            
            # 更新指标 (使用 detach() 释放显存)
            with torch.no_grad():
                metric.update(output.logits.detach(), labels, loss.detach())
            
            # 反向传播 (DeepSpeed 内部处理梯度累积)
            accelerator.backward(loss)

            # 优化器步进
            # 在 DeepSpeed 模式下，这些函数内部会自动判断是否达到了累积步数
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            acc, train_loss = metric.get_metric(reset=(global_step % 10 == 0))

            if accelerator.is_main_process:
                train_iter.set_postfix(epoch=epoch, loss=round(train_loss, 4), acc=round(acc, 4), lr=lr_scheduler.get_last_lr()[0])
                if global_step % 5 == 0:
                    wandb.log({'loss': train_loss, 'acc': acc, 'lr': lr_scheduler.get_last_lr()[0]}, step=global_step)

        save_checkpoint(epoch, batch_cnt, global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Experiment Args
    parser.add_argument('--experiment_name', type=str,default='sft_stage1')

    # Model Args
    parser.add_argument('--model_path', required=True, type=str)

    # Data Args
    parser.add_argument('--data_path', required=True, type=str)

    # Training Args
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=2, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=3, type=int)

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           

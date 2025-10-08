import os
import json
import torch
import logging
import argparse
import shutil
import traceback

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
import wandb
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        with open(config.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # optional simple copy/filter step preserved
        newdata = []
        for da in self.data:
            newdata.append(da)
        logger.info(f'加载数据 items: {len(self.data)} -> {len(newdata)}')
        self.data = newdata

        self.max_seq_len = self.config.max_seq_len
        self.debug = 0

        # Do NOT overwrite tokenizer.chat_template for Qwen3.
        # Use whatever the tokenizer provides. Build Template fallback only if chat_template exists as string.
        self.template = None
        if getattr(tokenizer, "chat_template", None):
            try:
                self.template = Template(tokenizer.chat_template)
            except Exception:
                self.template = None

    def __getitem__(self, index):
        return self.data[index]

    def get_response(self, da):
        # keep the same formatting you used previously
        temp = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        return temp.format(da.get('Complex_CoT', ''), da.get('Response', ''))

    def get_prompt(self, da):
        q = da.get('Question', '')
        a = self.get_response(da)
        assert q is not None and a is not None, f'q:{q} a:{a}'

        messages_full = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages_query = [{"role": "user", "content": q}]

        # Prefer tokenizer.apply_chat_template when available (Qwen tokenizer may provide helper).
        if hasattr(self.tokenizer, "apply_chat_template"):
            # defensive: different tokenizer versions may have different arg names.
            try:
                input_ids = self.tokenizer.apply_chat_template(messages_full, tokenize=True, add_generation_prompt=False)
                query_ids = self.tokenizer.apply_chat_template(messages_query, tokenize=True, add_generation_prompt=True)
            except TypeError:
                # fallback if named args differ
                input_ids = self.tokenizer.apply_chat_template(messages_full, True, False)
                query_ids = self.tokenizer.apply_chat_template(messages_query, True, True)
            # ensure lists of ints
            input_ids = list(input_ids)
            query_ids = list(query_ids)
        else:
            # fallback to using chat_template string and jinja rendering and tokenizer.encode
            if self.template is None:
                # last resort: simple concatenation with bos and eos.
                bos = getattr(self.tokenizer, "bos_token", "")
                sep = getattr(self.tokenizer, "eos_token", "")
                query = f"{bos}{q}{sep}"
                full = f"{bos}{q}{a}{sep}"
                query_ids = self.tokenizer.encode(query, add_special_tokens=False)
                input_ids = self.tokenizer.encode(full, add_special_tokens=False)
            else:
                input_text = self.template.render(messages=messages_full, bos_token=self.tokenizer.bos_token, add_generation_prompt=False)
                query_text = self.template.render(messages=messages_query, bos_token=self.tokenizer.bos_token, add_generation_prompt=True)
                input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
                query_ids = self.tokenizer.encode(query_text, add_special_tokens=False)

        # build labels mask: mask query part with -100
        labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
        # truncation to max_seq_len
        input_ids = input_ids[-self.max_seq_len:]
        labels = labels[-self.max_seq_len:]
        assert len(input_ids) == len(labels)
        return {"input_ids": input_ids, "labels": labels}

    def collate_fn(self, batch):
        data = [self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len, self.max_seq_len)
        input_ids = [item[:max_len] + [self.tokenizer.eos_token_id] * (max_len - len(item)) for item in input_ids]
        labels = [item[:max_len] + [-100] * (max_len - len(item)) for item in labels]

        if self.debug < 3:
            try:
                logger.info("example input decoded: %s", self.tokenizer.decode(input_ids[-1], skip_special_tokens=False))
                logger.info("example labels decoded: %s", self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            except Exception:
                pass
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
        # world_size must exist in distributed env
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            # compute token-level accuracy on shifted preds
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            # equality mask ignoring -100
            mask = shift_labels != -100
            self.right += (shift_preds == shift_labels).masked_fill(~mask, 0).sum().item()
            self.total += mask.sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        # reduce across processes if distributed
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item() if self.total.item() > 0 else 0.0
        loss = self.total_loss.item() / (self.world_size * max(1, self.n_step))

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss


def train(args):
    # Accelerator: bf16 recommended for Qwen3-8B
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, config=vars(args), dir=args.log_dir, mode="offline")

    accelerator.print(f'args:\n{args}')

    # update deepspeed config batch sizes if present
    try:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
        accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu * (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1) * accelerator.gradient_accumulation_steps
    except Exception:
        pass

    # load tokenizer and model (trust_remote_code True for Qwen)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # enable gradient checkpointing if model supports it
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = Train_dataset(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    # compute number of update steps
    effective_world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // max(1, effective_world_size)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=max(1, num_training_steps))
    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_path:{args.data_path} lr:{args.learning_rate} num_training_steps:{num_training_steps}')

    # prepare with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    start_epoch = 0
    start_step = 0
    global_step = 0

    metric = SFTMetric(device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

    def save_checkpoint(epoch, step, global_step):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, 'tfmr')

        # unwrap model and use accelerator.save to ensure all sharded weights are gathered correctly
        unwrap_model = accelerator.unwrap_model(model)
        # save_pretrained with accelerator.save and state_dict ensures DeepSpeed-Zero3 compatibility
        try:
            unwrap_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=True
            )
        except TypeError:
            # older HF versions may not accept state_dict arg in save_pretrained
            unwrap_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                safe_serialization=True
            )

        # tokenizer
        tokenizer.save_pretrained(output_dir)

        # save training state
        accelerator.wait_for_everyone()
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint saved to {output_dir}')

    accelerator.print("deepspeed config (if present):")
    try:
        accelerator.print(accelerator.deepspeed_config)
    except Exception:
        pass

    model.train()

    for epoch in range(start_epoch, args.n_epochs):
        train_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_iterator:
            if epoch == start_epoch and batch_cnt < start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids = batch['input_ids'].to(accelerator.device)
            labels = batch['labels'].to(accelerator.device)

            output = model(input_ids=input_ids, labels=labels, return_dict=True, use_cache=False)
            loss = output.loss

            # update metric
            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()

            accelerator.backward(loss)
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=input_ids.shape[1], lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

        accelerator.wait_for_everyone()
        # save at epoch end
        try:
            save_checkpoint(epoch, batch_cnt, global_step)
        except Exception as e:
            accelerator.print("save_checkpoint failed:", e)
            traceback.print_exc()

    accelerator.print("training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    # Experiment Args
    parser.add_argument('--experiment_name', type=str, default='sft_stage1')

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
    parser.add_argument('--train_bsz_per_gpu', default=2, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=3, type=int)

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)

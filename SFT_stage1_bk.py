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

os.umask(0)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        with open(config.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} records from {config.data_path}")
        self.max_seq_len = config.max_seq_len
        self.debug = 0

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

    def get_response(self, da):
        return '## Thinking\n\n{}\n\n## Final Response\n\n{}'.format(da.get('Complex_CoT',''), da.get('Response',''))

    def get_prompt(self, da):
        q = da.get('Question','')
        a = self.get_response(da)
        assert q is not None and a is not None
        messages_full = [{"role":"user","content":q},{"role":"assistant","content":a}]
        messages_query = [{"role":"user","content":q}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                input_ids = list(self.tokenizer.apply_chat_template(messages_full, tokenize=True, add_generation_prompt=False))
                query_ids = list(self.tokenizer.apply_chat_template(messages_query, tokenize=True, add_generation_prompt=True))
            except TypeError:
                input_ids = list(self.tokenizer.apply_chat_template(messages_full, True, False))
                query_ids = list(self.tokenizer.apply_chat_template(messages_query, True, True))
        else:
            # fallback simple concat
            full_text = q + "\n" + a
            query_text = q
            input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            query_ids = self.tokenizer.encode(query_text, add_special_tokens=False)

        labels = [-100]*len(query_ids) + input_ids[len(query_ids):]
        input_ids = input_ids[-self.max_seq_len:]
        labels = labels[-self.max_seq_len:]
        return {"input_ids": input_ids, "labels": labels}

    def collate_fn(self, batch):
        items = [self.get_prompt(d) for d in batch]
        input_ids = [it["input_ids"] for it in items]
        labels = [it["labels"] for it in items]
        max_len = min(max(len(x) for x in input_ids), self.max_seq_len)
        pad_id = self.tokenizer.eos_token_id if getattr(self.tokenizer, "eos_token_id", None) is not None else 0
        input_ids = [x[:max_len] + [pad_id] * (max_len - len(x)) for x in input_ids]
        labels = [x[:max_len] + [-100] * (max_len - len(x)) for x in labels]
        if self.debug < 3:
            try:
                logger.info("Decoded input example: %s", self.tokenizer.decode(input_ids[-1], skip_special_tokens=False))
            except Exception:
                pass
            self.debug += 1
        return {"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(labels)}

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.tensor([0.], device=device)
        self.total = torch.tensor([0.], device=device)
        self.total_loss = torch.tensor([0.], device=device)
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def __call__(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            mask = shift_labels != -100
            self.right += (shift_preds == shift_labels).masked_fill(~mask, 0).sum().item()
            self.total += mask.sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
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

def rotate_checkpoints(output_base, max_ckpts):
    if max_ckpts <= 0: 
        return
    ckpts = [os.path.join(output_base, d) for d in os.listdir(output_base) if d.startswith("checkpoint-")]
    ckpts = [d for d in ckpts if os.path.isdir(d)]
    if len(ckpts) <= max_ckpts:
        return
    ckpts.sort(key=lambda x: os.path.getctime(x))
    remove_cnt = len(ckpts) - max_ckpts
    for i in range(remove_cnt):
        try:
            shutil.rmtree(ckpts[i])
            logger.info("Removed old checkpoint %s", ckpts[i])
        except Exception:
            logger.warning("Failed to remove %s", ckpts[i])

def verify_checkpoint_files(save_dir):
    # quick integrity check: ensure tfmr dir exists and contains model files
    tfmr = os.path.join(save_dir, "tfmr")
    if not os.path.isdir(tfmr):
        return False, "tfmr folder missing"
    # look for safetensors or pytorch files
    found = False
    for f in os.listdir(tfmr):
        if f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".index.json"):
            found = True
            break
    if not found:
        return False, "no model weight files found in tfmr"
    # check config and tokenizer
    if not os.path.isfile(os.path.join(tfmr, "config.json")):
        return False, "config.json missing"
    return True, "ok"

def train(args):
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps)
    if accelerator.is_main_process:
        wandb.init(project=args.experiment_name, config=vars(args), dir=args.log_dir, mode="offline")
    accelerator.print(vars(args))

    try:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
        accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu * (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1) * accelerator.gradient_accumulation_steps
    except Exception:
        pass

    # If user passed a checkpoint root, prefer its tfmr subdir
    model_load_path = args.model_path
    if os.path.isdir(os.path.join(args.model_path, "tfmr")):
        model_load_path = os.path.join(args.model_path, "tfmr")

    tokenizer = AutoTokenizer.from_pretrained(model_load_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_load_path, trust_remote_code=True)

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = TrainDataset(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    eff_world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // max(1, eff_world)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * max(1, num_training_steps)), num_training_steps=max(1, num_training_steps))
    accelerator.print(f"num_training_steps: {num_training_steps}")

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    metric = SFTMetric(device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    global_step = 0
    start_epoch = 0
    start_step = 0

    def save_checkpoint(epoch, step, global_step):
        save_root = args.output_dir
        os.makedirs(save_root, exist_ok=True)
        save_dir = os.path.join(save_root, f"checkpoint-{epoch}-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, "tfmr")

        unwrap_model = accelerator.unwrap_model(model)
        # Save with accelerator.save to gather shards. Use safe_serialization.
        try:
            unwrap_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=True
            )
        except TypeError:
            # older hf versions may not accept state_dict
            unwrap_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                safe_serialization=True
            )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

        accelerator.wait_for_everyone()
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))

        # rotate checkpoints on main process
        if accelerator.is_main_process:
            rotate_checkpoints(args.output_dir, args.max_ckpts)
            ok, msg = verify_checkpoint_files(save_dir)
            if not ok:
                logger.error("Checkpoint integrity check failed: %s", msg)
                # leave files for debugging but warn
            else:
                logger.info("Checkpoint saved and verified at %s", output_dir)
        accelerator.print(f"Saved checkpoint {save_dir}")

    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_idx, batch in iterator:
            if epoch == start_epoch and batch_idx < start_step:
                continue
            if batch_idx == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids = batch['input_ids'].to(accelerator.device)
            labels = batch['labels'].to(accelerator.device)

            output = model(input_ids=input_ids, labels=labels, return_dict=True, use_cache=False)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()

            accelerator.backward(loss)
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                iterator.set_postfix(epoch=epoch, step=batch_idx, loss=round(train_loss,4), acc=round(acc,4), lr=lr_scheduler.get_last_lr()[0])

            if global_step % args.log_interval == 0 and accelerator.is_main_process:
                wandb.log({"loss": train_loss, "acc": acc, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

        accelerator.wait_for_everyone()
        try:
            save_checkpoint(epoch, batch_idx, global_step)
        except Exception as e:
            logger.exception("Save checkpoint failed: %s", e)

    accelerator.print("Training finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='sft_qwen3')
    parser.add_argument('--model_path', required=True, type=str)  # base model or checkpoint (root or .../tfmr)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=2, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=2, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=3, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)

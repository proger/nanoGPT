"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run in debug mode example:
$ python train.py --batch_size=32 --other=args

To run DDP on 4 gpus on one node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import sys
import time
import math
from ast import literal_eval

import wandb
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import bitsandbytes as bnb

from model import GPTConfig, GPT
from utils import optimizer_to

#torch.set_default_dtype(torch.bfloat16)

old_locals = set(locals())
# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = 'exp/gpt2-medium'
eval_interval = 5000
log_interval = 100
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
# wandb logging
wandb_log = True # disabled by default
wandb_entity = 'stud76'
wandb_project = 'ubertext'
wandb_run_name = 'gpt2-medium' # 'run' + str(time.time())
# data
dataset = 'uk4b'
# aim for 0.5M tokens per gradient update
batch_size = 4
block_size = 1024
crop_block_size = 1024
grad_acc_steps = 128
seed_base = 0
# model
device = 'cuda:0'
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
n_layer = 24
n_head = 16
n_embd = 1024
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 1<<20 # total number of training iterations
weight_decay = 1e-2
betas = (0.9, 0.95)
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
warmup_iters = 2000*grad_acc_steps # how many steps to warm up for
lr_decay_iters = max_iters - warmup_iters # how many steps to decay the learning rate for
min_lr = 6e-5 # minimum learning rate
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
compile_model = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# poor man's Configurator. Potentially a bad idea. Example usage:
# $ python train.py override_file --batch_size=32
# this will first run config/override_file.py, then override batch_size to 32
def configure():
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = os.path.join('config', arg + '.py')
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read())
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            if key in globals():
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(globals()[key])
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                globals()[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
configure()
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
else:
    gpu_id = 0 # gpu_id 0 means this is the (single) master process, basically

run_config = {k: globals()[k] for k in locals() if k not in old_locals and k != 'configure' and k != 'old_locals'}

if gpu_id == 0:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed_base + gpu_id) # note: each worker gets a different seed
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.set_float32_matmul_precision('high')

# poor man's data loader, TODO evaluate need for actual DataLoader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer = n_layer, n_head = n_head, n_embd = n_embd, block_size = block_size, dropout = dropout)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    for k, v in model_args.items():
        if k in ("block_size", "dropout"):
            continue
        assert checkpoint_model_args[k] == v, f"{k} mismatching"

    gptconf = GPTConfig(**model_args)

    if compile_model:
        # model compilation prefixes all keys with _orig_mod
        model = torch.nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'], strict=False)
        model = model['_orig_mod']
    else:
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'], strict=False)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    model = GPT.from_pretrained(init_from)


# crop down the model block size if desired
if block_size < model.block_size:
    print('cropping model block size from', model.block_size, 'to', block_size)
    model.crop_block_size(crop_block_size)
model.to(device)

@torch.no_grad()
def estimate_loss(eval=True):
    out = {}
    if eval:
        model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



# compile the model
if compile_model:
    print("compiling the model... (takes a ~minute)")
    #unoptimized_model = model
    model = torch.compile(model, mode="default") # requires PyTorch 2.0

    #for x in range(1):
    #    print('compiling?', estimate_loss(eval=False)) # dummy forward

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer_to(optimizer, device)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[gpu_id])


# learning rate decay scheduler (cosine with warmup)
def get_lr(iter):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

run_config.update({
    "crop_block_size": crop_block_size,
    "tokens_per_step": batch_size*crop_block_size*grad_acc_steps,
})
print('run_config = ', run_config, flush=True)
# logging
if wandb_log and gpu_id == 0:
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config=run_config)

# training loop
t0 = time.time()
optimizer.zero_grad(set_to_none=True)
while True:

    # determine the learning rate for this iteration
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = learning_rate

    if iter_num % eval_interval == 0 and gpu_id == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            }, step=iter_num)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            raw_model = model.module if ddp else model
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    X, Y = get_batch('train')
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(X, Y)

    loss.backward()
    # TODO: gradient clipping evaluate need for
    if iter_num and iter_num % grad_acc_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and gpu_id == 0:
        lossf = loss.item() # loss as float. TODO CPU-GPU sync: profile, make sure not slow af
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
            }, step=iter_num)
    iter_num += 1

    # termination conditions
    if iter_num >= max_iters:
        break

destroy_process_group()

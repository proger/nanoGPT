"""
Sample from a trained model
"""
import argparse
import os
import torch
import torch.nn as nn
from model import GPTConfig, GPT
import sentencepiece as spm
import sys

parser = argparse.ArgumentParser('sample')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--no_eot', action='store_true')
parser.add_argument('ckpt_path')
parser.add_argument('prompts', nargs='+')
args = parser.parse_args()

device = args.device
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

checkpoint = torch.load(args.ckpt_path, map_location=device)

# model
gptconf = GPTConfig(**checkpoint['model_args'])
model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)
model = model._orig_mod
model = torch.compile(model) # requires PyTorch 2.0

sp = spm.SentencePieceProcessor(model_file='data/uk8/uk8_gpt.model')

for prompt in args.prompts:
    if args.no_eot:
        start = sp.encode(prompt)
    else:
        start = [50256] + sp.encode(prompt)
    x = (torch.tensor(start, dtype=torch.long, device=device)[None, ...])

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = model.generate(x, 50, temperature=0.9, top_k=100)

    print(sp.decode(y[0].tolist()))
    print('---------------')

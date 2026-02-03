#!/usr/bin/env python3
import os
os.environ["TINYGRAD_DEVICE"] = "CPU"
import gc

from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from newcontrol import get_foundation
from video import frames_from_file

Tensor.training = False
Tensor.no_grad = True
import numpy as np
from tqdm import tqdm, trange

def preprocess(foundation, fn, fn_out, flip=False, batch_size=64):
  if not fn.endswith(".hevc"): return
  if 'online' in fn: return
  print("preprocessing", fn, fn_out)
  if os.path.isfile(fn_out): return
  frms = []
  xs = []
  for frm in tqdm(frames_from_file(fn)):
    frms.append(frm)
    if len(frms) == 256:
      data = Tensor(np.concatenate([x.reshape(1,480,640,3) for x in frms]))
      if flip: data = data[:, :, ::-1]
      xs.append(foundation(data).realize())
      frms = []
  if frms:
    data = Tensor(np.concatenate([x.reshape(1,480,640,3) for x in frms]))
    if flip: data = data[:, :, ::-1]
    xs.append(foundation(data).realize())
    
    for i in trange(0, len(data), batch_size):        #batch size required to not load entirely on gpu
        batch = data[i:i+batch_size]
        feats = foundation(batch).realize()
        xs.append(feats)

  big_x = Tensor.cat(*xs, dim=0)
  print(f"Saving {fn_out}, shape: {big_x.shape}")
  #big_x=big_x.cpu()
  safe_save({"x": big_x}, fn_out)
  del big_x, xs, data                 #delete load 
  gc.collect()

if __name__ == "__main__":
  foundation = get_foundation()
  for fn in os.listdir("data"):
    preprocess(foundation, "data/"+fn, "data/"+fn.replace(".hevc", ".safetensors"))
    preprocess(foundation, "data/"+fn, "data/"+fn.replace(".hevc", "_flip.safetensors"), True)


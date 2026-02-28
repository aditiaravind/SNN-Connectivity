import subprocess
import psutil
import gc
from spikingjelly.activation_based.neuron import ParametricLIFNode as PLIFNode
from spikingjelly.activation_based.neuron import GatedLIFNode as GLIFNode
from spikingjelly.activation_based.neuron import LIFNode
# import torch
import glob
import h5py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import logging

def new_gen(keys, filename, model, root_folder = '/mnt/data/models', attr=False, check_files = False):
    file = glob.glob(os.path.join(root_folder,  '*'+model+'*', '*'+filename+'*'))[0]
    if check_files:
        print(file)
    for key in keys:
        with h5py.File(file, 'r') as f:
            yield np.array(f[key])

    
def setup_logger(log_path, level=logging.INFO):
    
    logger = logging.getLogger(log_path)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
    
class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
        
def track_gpu_mem(gpu_id = None, clear=False):
    # if clear:
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()
    #     gc.collect()

    mem = psutil.virtual_memory()
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
         '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )

    lines = result.strip().split('\n')
    
    if gpu_id is not None:
        _, _, free = map(int, lines[gpu_id].split(','))
        print(f">>>> GPU: {free/1024:.2f} GB, CPU: {mem.available / 1e9:.2f} GB")
        
    else:    
        for i, line in enumerate(lines):
            total, used, free = map(int, line.split(','))
            print(f"GPU {i}: Free: {free/1024:.2f} GB / Total: {total/1024:.2f} GB")
        print(f"CPU RAM Available: {mem.available / 1e9:.2f} GB")


def log_tau_after_epoch(model, tracked_taus):
    for name, module in model.named_modules():
        if isinstance(module, PLIFNode):
            if name not in tracked_taus:
                tracked_taus[name] = []
            w_temp = module.w.sigmoid().clone().detach().cpu().numpy()
            tracked_taus[name].append(1./w_temp)



class ImagenetteDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        # Collect (image_path, label_index) pairs
        # self.classes = sorted(os.listdir(self.root_dir))
        self.classes = sorted([d for d in os.listdir(self.root_dir) 
                       if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_path = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls_name]))
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


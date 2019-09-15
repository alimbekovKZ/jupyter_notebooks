import cv2
import os
import torch
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
SEED = 2019
seed_everything(SEED)

cv2.setNumThreads(0)  # fix potential pytorch worker issues

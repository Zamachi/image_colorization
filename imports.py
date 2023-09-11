from PIL import Image
import numpy as np
from skimage.color import rgb2lab as rgb2lab, lab2rgb as lab2rgb, rgb2gray as rgb2gray
from skimage.io import imsave as imsave
import torch
import math, time
from tqdm.auto import tqdm as tqdm, trange as trange
from lion_pytorch import Lion as Lion
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader as DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance as FrechetInceptionDistance
from gc import collect as collect
from os.path import isfile as isfile, exists as exists
from os import remove as remove, makedirs as makedirs 
from psutil import virtual_memory as virtual_memory
from torchvision import transforms as transforms
import zipfile 
from itertools import islice as islice
# import kaggle.api
from huggingface_hub import notebook_login as notebook_login
from torch.nn.functional import cross_entropy as cross_entropy
from scipy.ndimage import gaussian_filter1d as gaussian_filter1d
from warnings import warn as warn
import matplotlib as plt
# NOTE: moji moduli

dataset_path='./dataset/'
model_weights_path = './model_weights/'
# try:
#     import google.colab
#     from google.colab import output
#     output.enable_custom_widget_manager()
#     print("u kolabu")
#     IN_COLAB = True
# except:
#     IN_COLAB = False

# if IN_COLAB:
#     %pip install torchvision torchaudio torch scipy numpy scikit-image lion-pytorch torchmetrics


seed=42
_ = torch.manual_seed(seed)
_np = np.random.seed(seed)

if torch.cuda.is_available():
    _cuda = torch.cuda.manual_seed_all(seed)
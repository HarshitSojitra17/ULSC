import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from similarity import runs
from psnr import runpsnr
import pandas as pd

# Configure GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    SM = []
    PSNR = []
    sm = runs()
    psnr = runpsnr()
    SM.append(sm)
    PSNR.append(psnr)
    print('sm:', np.mean(SM))
    print('psnr:', np.mean(PSNR))




if __name__ == '__main__':
    main()
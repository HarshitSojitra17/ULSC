import torch
from torchvision.transforms import ToTensor
from PIL import Image
import math
import numpy as np
import os


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return ToTensor()(image)


def calculate_psnr(image1, image2):
    mse = torch.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 10 * math.log10(max_pixel**2 / mse)
    return psnr


def runpsnr():
    PSNR = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resize_dir = os.path.join(base_dir, '..', 'testdata', 'resizedata')
    output_dir = os.path.join(base_dir, 'outputdata')

    for i in range(96, 101):
        image_path1 = os.path.join(resize_dir, f'image_{i}.png')
        image_path2 = os.path.join(output_dir, f'image_image_{i}.png')

        image1 = load_image(image_path1)
        image2 = load_image(image_path2)

        psnr_value = calculate_psnr(image1, image2)
        print(f'  Image {i} PSNR: {psnr_value:.2f} dB')
        PSNR.append(psnr_value)
    return float(np.mean(PSNR))


if __name__ == "__main__":
    result = runpsnr()
    print("Final PSNR:", result)
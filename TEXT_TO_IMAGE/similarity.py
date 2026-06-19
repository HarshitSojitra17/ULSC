import torch
import clip
import torch
from PIL import Image
import numpy as np
import os

# Load pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Image preprocessing
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)
    return image


# Calculate semantic similarity between two images
def calculate_semantic_similarity(image_path1, image_path2, model):
    image1 = load_and_preprocess_image(image_path1)
    image2 = load_and_preprocess_image(image_path2)

    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    similarity = torch.cosine_similarity(image_features1, image_features2)
    return similarity.item()


def runs():
    # 示例
    sm=[]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resize_dir = os.path.join(base_dir, '..', 'testdata', 'resizedata')
    output_dir = os.path.join(base_dir, 'outputdata')
    for i in range(96, 101):
        image_path1 = os.path.join(resize_dir, f'image_{i}.png')
        image_path2 = os.path.join(output_dir,f'image_image_{i}.png')
        similarity = calculate_semantic_similarity(image_path1, image_path2, model)
        sm.append(similarity)
        # print(f'{i}图片之间的语义相似度为: {similarity}')
    return np.mean(sm)


if __name__  == "__main__":
    result = runs()
    print("Final Similarity:", result)

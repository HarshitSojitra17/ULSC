

import os
from PIL import Image

# 定义图像文件夹路径、目标尺寸和保存文件夹路径
input_folder_path = './004_1'
output_folder_path = './outputdata'
target_size = (256, 256)

# 创建保存文件夹
os.makedirs(output_folder_path, exist_ok=True)

# 遍历文件夹中的每个图像文件，并逐张处理
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 构建图像文件的完整路径
        image_path = os.path.join(input_folder_path, filename)

        # 打开图像文件
        image = Image.open(image_path)

        # 调整图像尺寸
        resized_image = image.resize(target_size)

        # 构建保存的文件路径
        output_image_path = os.path.join(output_folder_path, f"image_{filename[:-4]}.png")

        # 保存调整尺寸后的图像
        resized_image.save(output_image_path)

print("图像 resize 并保存完成！")
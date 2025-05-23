import os
import shutil

# 定义文件夹路径
gt_folder = r"E:\CODE\image_retouching\Exposure_Correction-pytorch-main\MultiExposure_dataset\testing\gt_c\512"  # gt文件夹路径
input_folder = r"E:\CODE\image_retouching\Exposure_Correction-pytorch-main\MultiExposure_dataset\testing\INPUT_IMAGES"  # 原始input文件夹路径
new_input_folder = r"E:\CODE\image_retouching\Exposure_Correction-pytorch-main\MultiExposure_dataset\testing\input_p1/exp1"  # 新的input文件夹路径

# 创建新的input文件夹
if not os.path.exists(new_input_folder):
    os.makedirs(new_input_folder)

# 获取gt文件夹中的所有图片名称
gt_images = os.listdir(gt_folder)

# 遍历gt文件夹中的每张图片
for gt_image in gt_images:
    # 获取不带扩展名的图片名称
    image_name = os.path.splitext(gt_image)[0]  # 去掉扩展名（.jpg）

    # 构造对应的以 "_0" 结尾的input图片名称
    input_image_name = f"{image_name}_P1.JPG"  # 假设扩展名是大写的.JPG

    # 检查该图片是否存在
    if input_image_name in os.listdir(input_folder):
        # 构造源文件路径和目标文件路径
        src_path = os.path.join(input_folder, input_image_name)
        dst_path = os.path.join(new_input_folder, gt_image)  # 使用与gt图片相同的文件名

        # 复制文件
        shutil.copy(src_path, dst_path)
        print(f"Copied {input_image_name} to {dst_path}")
    else:
        print(f"No input image found for {gt_image} with '_0.JPG' suffix")

print("New input folder created with images ending with '_0'.")
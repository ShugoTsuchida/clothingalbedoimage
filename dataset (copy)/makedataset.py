import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

def extract_clothing_region(image):
    """服領域を抽出するためのセグメンテーション"""
    segmentation_model = deeplabv3_resnet50(pretrained=True)
    segmentation_model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]

    clothing_classes = [5, 6, 7, 9, 10, 11, 14, 15]  # セグメンテーションの服に該当するクラスID
    mask = torch.zeros_like(output[0], dtype=torch.bool)
    for cls_id in clothing_classes:
        mask = mask | (output.argmax(0) == cls_id)

    mask = mask.cpu().numpy().astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def crop_upper_body(image, mask):
    """上半身領域を切り出す"""
    height, width = image.shape[:2]

    # 上部20%カット、下部50%カット、左右20%カット
    top_cut = int(height * 0.20)
    bottom_cut = int(height * 0.50)
    side_cut = int(width * 0.20)

    upper_body_x1 = side_cut
    upper_body_x2 = width - side_cut
    upper_body_y1 = top_cut
    upper_body_y2 = height - bottom_cut

    cropped_image = image[upper_body_y1:upper_body_y2, upper_body_x1:upper_body_x2]
    cropped_mask = mask[upper_body_y1:upper_body_y2, upper_body_x1:upper_body_x2]

    return cropped_image, cropped_mask

def process_images(input_dir, output_dir):
    """データセットを作成するための画像処理"""
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: {input_path} が読み込めません")
            continue

        # セグメンテーションで服領域を抽出
        clothing_mask = extract_clothing_region(image)

        # 上半身領域を切り出し
        cropped_image, cropped_mask = crop_upper_body(image, clothing_mask)

        # マスク適用
        result_image = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

        # 保存
        cv2.imwrite(output_path, result_image)

if __name__ == "__main__":
    input_dir = "/home/tsuchida/unsupervised-learning-intrinsic-images/ShadingNet/croped_dataset/03/white/0"
    output_dir = "/home/tsuchida/unsupervised-learning-intrinsic-images/ShadingNet/croped_dataset/03/white/onlyclothing"


    process_images(input_dir, output_dir)

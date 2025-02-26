import sys, traceback
from datetime import datetime
import os
import torch
import cv2
import numpy as np

import pandas as pd
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import color
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.models import create_model
from options.test_options import TestOptions

# モデルの初期化（元のコードと同じ）
opt = TestOptions().parse()
opt.isTrain = False
opt.lr = 0
opt.checkpoints_dir = './checkpoints/'
opt.name = 'test_local'

model = create_model(opt)
weight_path = os.path.join(opt.checkpoints_dir, opt.name, 'paper_final_net_G.pth')
state_dict = torch.load(weight_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.netG.load_state_dict(state_dict)
model.switch_to_eval()

yolo_model = YOLO('yolov5s.pt')
segmentation_model = deeplabv3_resnet50(pretrained=True)
segmentation_model.eval()

def calculate_color_stats(image, mask=None):
    """
    RGB と HSV の統計値を scikit-image を使用して計算
    """
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    valid_pixels = np.any(rgb_image > 0, axis=2)
    
    rgb_norm = rgb_image.astype(float) / 255.0
    r_channel = rgb_image[valid_pixels, 0]
    g_channel = rgb_image[valid_pixels, 1]
    b_channel = rgb_image[valid_pixels, 2]
    
    # RGB統計
    rgb_stats = {
        'r_mean': np.mean(r_channel) if r_channel.size > 0 else 0,
        'r_median': np.median(r_channel) if r_channel.size > 0 else 0,
        'g_mean': np.mean(g_channel) if g_channel.size > 0 else 0,
        'g_median': np.median(g_channel) if g_channel.size > 0 else 0,
        'b_mean': np.mean(b_channel) if b_channel.size > 0 else 0,
        'b_median': np.median(b_channel) if b_channel.size > 0 else 0
    }
    
    # HSV統計とヒストグラム用データ
    hsv_image = color.rgb2hsv(rgb_norm)
    hsv_valid = hsv_image[valid_pixels]
    
    # Hチャンネル（0-360度）の処理
    h_channel = (hsv_valid[:, 0] * 360).astype(int)  # 0-360度に変換
    h_mode = int(np.bincount(h_channel).argmax()) if h_channel.size > 0 else 0
    
    hsv_stats = {
        'h_mean': np.mean(h_channel) if h_channel.size > 0 else 0,
        'h_mode': h_mode,
        's_mean': np.mean(hsv_valid[:, 1]) * 100,
        'v_mean': np.mean(hsv_valid[:, 2]) * 100
    }
    
    # LAB統計とヒストグラム用データ
    lab_image = color.rgb2lab(rgb_norm)
    lab_valid = lab_image[valid_pixels]
    
    # Lチャンネル（0-100）の処理
    l_channel = lab_valid[:, 0]  # すでに0-100のスケール
    l_mode = int(np.bincount(l_channel.astype(int)).argmax()) if l_channel.size > 0 else 0
    
    lab_stats = {
        'l_mean': np.mean(l_channel) if l_channel.size > 0 else 0,
        'l_mode': l_mode,
        'a_mean': np.mean(lab_valid[:, 1]),
        'b_lab_mean': np.mean(lab_valid[:, 2])
    }
    
    stats = {**rgb_stats, **hsv_stats, **lab_stats}
    return stats, r_channel, g_channel, b_channel, h_channel, l_channel

def save_rgb_histograms(r_channel, g_channel, b_channel, output_dir, frame_num, label="input"):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 3, 1)
    plt.hist(r_channel.ravel(), bins=256, color='red', alpha=0.7)
    plt.title(f'Red Channel ({label})')
    plt.xlim(0, 255)
    
    plt.subplot(1, 3, 2)
    plt.hist(g_channel.ravel(), bins=256, color='green', alpha=0.7)
    plt.title(f'Green Channel ({label})')
    plt.xlim(0, 255)
    
    plt.subplot(1, 3, 3)
    plt.hist(b_channel.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title(f'Blue Channel ({label})')
    plt.xlim(0, 255)
    
    plt.tight_layout()
    hist_filename = os.path.join(output_dir, f"frame_{frame_num:04d}_{label}_rgb_histogram.png")
    plt.savefig(hist_filename)
    plt.close()

def save_h_l_histograms(h_channel, l_channel, output_dir, frame_num, label="input"):
    """HとLチャンネルのヒストグラムを保存"""
    plt.figure(figsize=(12, 5))
    
    # Hチャンネルのヒストグラム（0-360度）
    plt.subplot(1, 2, 1)
    plt.hist(h_channel, bins=360, range=(0, 360), color='orange', alpha=0.7)
    plt.title(f'Hue Distribution ({label})')
    plt.xlabel('Hue Angle (degrees)')
    plt.ylabel('Frequency')
    
    # Lチャンネルのヒストグラム（0-100）
    plt.subplot(1, 2, 2)
    plt.hist(l_channel, bins=100, range=(0, 100), color='gray', alpha=0.7)
    plt.title(f'Lightness Distribution ({label})')
    plt.xlabel('Lightness Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    hist_filename = os.path.join(output_dir, f"frame_{frame_num:04d}_{label}_h_l_histogram.png")
    plt.savefig(hist_filename)
    plt.close()

def save_color_stats_to_csv(output_dir, frame_num, color_stats, stat_type):
    csv_filename = os.path.join(output_dir, f"frame_{frame_num:04d}_{stat_type}_color_stats.csv")
    pd.DataFrame([color_stats]).to_csv(csv_filename, index=False)

def decompose_image(input_image):
    """画像全体を反射成分に分解する"""
    height, width = input_image.shape[:2]
    target_size = opt.fineSize

    input_image_resized = cv2.resize(input_image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    input_image_tensor = input_image_resized / 255.0
    input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()

    prediction_S, prediction_R, _ = model.netG.forward(input_image_tensor)

    # Reflectance画像を正規化して保存
    prediction_R = prediction_R.cpu().detach().numpy()
    prediction_R = (prediction_R[0] - prediction_R.min()) / (prediction_R.max() - prediction_R.min()) * 255.0
    reflectance_image = prediction_R.transpose(1, 2, 0).astype('uint8')
    reflectance_image = cv2.resize(reflectance_image, (width, height), interpolation=cv2.INTER_LANCZOS4)

    return reflectance_image

def extract_clothing_region(image):
    """服領域を抽出する"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
    
    clothing_classes = [5, 6, 7, 9, 10, 11, 14, 15]
    mask = torch.zeros_like(output[0], dtype=torch.bool)
    for cls_id in clothing_classes:
        mask = mask | (output.argmax(0) == cls_id)
    
    mask = mask.cpu().numpy().astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def process_frame_decompose_first(input_image_path, output_dir, frame_num):
    """1フレームの処理（画像分解優先）"""
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: {input_image_path} が読み込めません")
        return

    # 1. まず画像全体を分解
    reflectance_image = decompose_image(input_image)

    # 2. セグメンテーションによる人物領域の抽出
    try:
        # マスクを適用
        clothing_mask = extract_clothing_region(input_image)
        clothing_original = cv2.bitwise_and(input_image, input_image, mask=clothing_mask)
        clothing_reflectance = cv2.bitwise_and(reflectance_image, reflectance_image, mask=clothing_mask)


        # 上半身領域の計算
        height, width = input_image.shape[:2]
        top_cut = int(height * 0.20)
        bottom_cut = int(height * 0.50)
        side_cut = int(width * 0.20)

        #10
        # height, width = input_image.shape[:2]
        # top_cut = int(height * 0.2525)
        # bottom_cut = int(height * 0.5025)
        # side_cut = int(width * 0.25)

        # #20
        # height, width = input_image.shape[:2]
        # top_cut = int(height * 0.2825)
        # bottom_cut = int(height * 0.505)
        # side_cut = int(width * 0.30)

        upper_body_x1 = side_cut
        upper_body_x2 = width - side_cut
        upper_body_y1 = top_cut
        upper_body_y2 = height - bottom_cut
        
        cropped_original = clothing_original[upper_body_y1:upper_body_y2, upper_body_x1:upper_body_x2]
        cropped_reflectance = clothing_reflectance[upper_body_y1:upper_body_y2, upper_body_x1:upper_body_x2]
        person_dir = os.path.join(output_dir, f"person_{frame_num:04d}")
        os.makedirs(person_dir, exist_ok=True)

        # 画像の保存
        cv2.imwrite(os.path.join(person_dir, f"frame_{frame_num:04d}_original.png"), cropped_original)
        cv2.imwrite(os.path.join(person_dir, f"frame_{frame_num:04d}_reflectance.png"), cropped_reflectance)

        # 統計データの保存
        original_stats, r_orig, g_orig, b_orig, h_orig, l_orig = calculate_color_stats(clothing_original, mask=clothing_mask)
        reflectance_stats, r_ref, g_ref, b_ref, h_ref, l_ref = calculate_color_stats(clothing_reflectance, mask=clothing_mask)

        # 統計データをCSVに保存
        save_color_stats_to_csv(person_dir, frame_num, original_stats, 'input')
        save_color_stats_to_csv(person_dir, frame_num, reflectance_stats, 'reflectance')

        # HとLのヒストグラムの保存
        save_h_l_histograms(h_orig, l_orig, person_dir, frame_num, "input")
        save_h_l_histograms(h_ref, l_ref, person_dir, frame_num, "reflectance")

    except Exception as e:
        print(f"Error processing frame {frame_num}: {e}")
        traceback.print_exc()

def process_sequence_decompose_first(image_sequence_dir, output_dir_base, max_frames=None, batch_size=50):
    """画像シーケンス全体の処理（分解優先アプローチ）"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir_base, f"output_decompose_first_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(os.listdir(image_sequence_dir))
    if max_frames:
        image_files = image_files[:max_frames]

    for batch_start in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[batch_start:batch_start+batch_size]

        for frame_num, image_file in enumerate(batch_files, start=batch_start):
            try:
                input_image_path = os.path.join(image_sequence_dir, image_file)
                frame_output_dir = os.path.join(output_dir, f"frame_{frame_num:04d}")
                os.makedirs(frame_output_dir, exist_ok=True)

                process_frame_decompose_first(input_image_path, frame_output_dir, frame_num)

                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                traceback.print_exc()
                continue

    print(f"処理完了: {output_dir}")

if __name__ == "__main__":
    image_sequence_dir = ""
    output_dir_base = ""

    process_sequence_decompose_first(
        image_sequence_dir,
        output_dir_base,
        max_frames=920, # 必要に応じて変更
        batch_size=50
    )

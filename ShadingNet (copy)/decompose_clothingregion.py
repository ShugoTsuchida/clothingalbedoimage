import sys, traceback
from datetime import datetime
import os
import tensorflow as tf
import cv2
import numpy as np

import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import color
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Parameters import Parameters
import ConvNets as cnn

# ShadingNetのパラメータ
image_height = 256
image_width = 256
image_channels = 3
batch_size = 1

sess = tf.Session()
I = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_height, image_width, image_channels], name='composite')
params = Parameters('first_pipeline')
weights = params.getWeight()
ambient_albedo, shadow_albedo, shading_albedo, albedo_reconstructed, shading_reconstructed, ambient_reconstructed, shadow_reconstructed = cnn.first_pipeline(I, weights, False)
direct_shading_reconstructed = shading_reconstructed - ambient_reconstructed + shadow_reconstructed
direct_shading_reconstructed = tf.nn.relu(direct_shading_reconstructed)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess, './checkpoints/shadingNet-440000')

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
def decompose_image_with_shadingnet(input_image, mask):
    """ShadingNetを使用して画像を分解 (非黒領域のみ)"""
    # マスク適用
    masked_image = cv2.bitwise_and(input_image, input_image, mask=mask)
    image_resized = cv2.resize(masked_image, (image_height, image_width)).astype('float32')
    batch_composite = image_resized[None, ...]
    feed_dict = {I: batch_composite}

    predictions_albedo, predictions_shading = sess.run([albedo_reconstructed, shading_reconstructed], feed_dict)
    albedo_image = cv2.resize(np.squeeze(predictions_albedo[0, ...]).astype('uint8'), (input_image.shape[1], input_image.shape[0]))
    shading_image = cv2.resize(np.squeeze(predictions_shading[0, ...]).astype('uint8'), (input_image.shape[1], input_image.shape[0]))

    # 分解結果にマスク適用
    albedo_image = cv2.bitwise_and(albedo_image, albedo_image, mask=mask)
    shading_image = cv2.bitwise_and(shading_image, shading_image, mask=mask)

    return albedo_image, shading_image

def process_frame_decompose_first(input_image_path, output_dir, frame_num):
    """1フレームの処理（ShadingNet使用版、非黒領域のみ処理）"""
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: {input_image_path} が読み込めません")
        return

    # 非黒領域のマスク作成
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # ShadingNetで画像を分解
    albedo_image, shading_image = decompose_image_with_shadingnet(input_image, mask)

    try:
        person_dir = os.path.join(output_dir, f"person_{frame_num:04d}")
        os.makedirs(person_dir, exist_ok=True)

        # 画像の保存
        cv2.imwrite(os.path.join(person_dir, f"frame_{frame_num:04d}_original.png"), input_image)
        cv2.imwrite(os.path.join(person_dir, f"frame_{frame_num:04d}_albedo.png"), albedo_image)
        cv2.imwrite(os.path.join(person_dir, f"frame_{frame_num:04d}_mask.png"), mask)

        # 統計データの保存
        original_stats, r_orig, g_orig, b_orig, h_orig, l_orig = calculate_color_stats(input_image, mask=mask)
        albedo_stats, r_alb, g_alb, b_alb, h_alb, l_alb = calculate_color_stats(albedo_image, mask=mask)

        # 統計データをCSVに保存
        save_color_stats_to_csv(person_dir, frame_num, original_stats, 'input')
        save_color_stats_to_csv(person_dir, frame_num, albedo_stats, 'albedo')

        # HとLのヒストグラムの保存
        save_h_l_histograms(h_orig, l_orig, person_dir, frame_num, "input")
        save_h_l_histograms(h_alb, l_alb, person_dir, frame_num, "albedo")

    except Exception as e:
        print(f"Error processing frame {frame_num}: {e}")
        traceback.print_exc()

def process_sequence_decompose_first(image_sequence_dir, output_dir_base, max_frames=None, batch_size=50):
    """画像シーケンス全体の処理（ShadingNet使用版）"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir_base, f"blackback")
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
import os
import torch
import cv2
import numpy as np
from datetime import datetime
import sys, traceback
import pandas as pd
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import color
from models.models import create_model
from options.test_options import TestOptions

os.environ['MPLBACKEND'] = 'Agg'

# 訓練オプションを読み込む
opt = TestOptions().parse()
opt.isTrain = False
opt.lr = 0
opt.checkpoints_dir = './checkpoints/'
opt.name = 'test_local'

# モデルのインスタンス化
model = create_model(opt)

# 学習済み重みのロード
weight_path = os.path.join(opt.checkpoints_dir, opt.name, 'paper_final_net_G.pth')
if not os.path.exists(weight_path):
    raise FileNotFoundError(f"Weight file not found: {weight_path}")

state_dict = torch.load(weight_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.netG.load_state_dict(state_dict)
print("Model weights loaded successfully!")

model.switch_to_eval()

def create_valid_pixel_mask(image):
    """
    黒い背景を除外するマスクを作成
    """
    # すべてのチャンネルが0のピクセルを特定
    return np.any(image > 0, axis=2)

def calculate_color_stats(image):
    """
    RGB と HSV の統計値を計算
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_norm = rgb_image.astype(float) / 255.0
    
    # チャンネルを分離
    r_channel = rgb_image[:, :, 0].ravel()
    g_channel = rgb_image[:, :, 1].ravel()
    b_channel = rgb_image[:, :, 2].ravel()
    
    # RGB統計
    rgb_stats = {
        'r_mean': np.mean(r_channel),
        'r_median': np.median(r_channel),
        'g_mean': np.mean(g_channel),
        'g_median': np.median(g_channel),
        'b_mean': np.mean(b_channel),
        'b_median': np.median(b_channel)
    }
    
    # HSV統計
    hsv_image = color.rgb2hsv(rgb_norm)
    h_channel = (hsv_image[:, :, 0].ravel() * 360).astype(int)
    h_mode = int(np.bincount(h_channel).argmax())
    
    hsv_stats = {
        'h_mean': np.mean(h_channel),
        'h_mode': h_mode,
        's_mean': np.mean(hsv_image[:, :, 1].ravel()) * 100,
        'v_mean': np.mean(hsv_image[:, :, 2].ravel()) * 100
    }
    
    # LAB統計
    lab_image = color.rgb2lab(rgb_norm)
    l_channel = lab_image[:, :, 0].ravel()
    l_mode = int(np.bincount(l_channel.astype(int)).argmax())
    
    lab_stats = {
        'l_mean': np.mean(l_channel),
        'l_mode': l_mode,
        'a_mean': np.mean(lab_image[:, :, 1].ravel()),
        'b_lab_mean': np.mean(lab_image[:, :, 2].ravel())
    }
    
    stats = {**rgb_stats, **hsv_stats, **lab_stats}
    return stats, r_channel, g_channel, b_channel, h_channel, l_channel

def save_h_l_histograms(h_channel, l_channel, output_dir, frame_num, label="input"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(h_channel, bins=360, range=(0, 360), color='orange', alpha=0.7)
    plt.title(f'Hue Distribution ({label})')
    plt.xlabel('Hue Angle (degrees)')
    plt.ylabel('Frequency')
    
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

def preprocess_image(image):
    """
    画像から有効なピクセル領域のみを抽出して前処理
    """
    # 有効なピクセルのマスクを作成
    valid_mask = create_valid_pixel_mask(image)
    
    # バウンディングボックスを取得
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # 有効な領域を切り出し
    valid_region = image[y_min:y_max+1, x_min:x_max+1]
    
    return valid_region, valid_mask

def process_image(input_image_path, output_dir, frame_num):
    """
    画像処理関数（黒い部分を事前に除外）
    """
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: {input_image_path} が読み込めませんでした")
        return False

    # 元の画像サイズを保存
    original_height, original_width = input_image.shape[:2]

    # 前処理（有効なピクセル領域の抽出）
    valid_region, valid_mask = preprocess_image(input_image)
    if valid_region.size == 0:
        print(f"Warning: No valid pixels in frame {frame_num}")
        return False

    # 統計データの保存（入力画像）
    original_stats, r_orig, g_orig, b_orig, h_orig, l_orig = calculate_color_stats(valid_region)
    save_color_stats_to_csv(output_dir, frame_num, original_stats, 'input')
    save_h_l_histograms(h_orig, l_orig, output_dir, frame_num, "input")

    # 有効領域をモデルの入力サイズにリサイズ
    target_size = opt.fineSize
    valid_region_resized = cv2.resize(valid_region, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # テンソルに変換
    input_tensor = valid_region_resized / 255.0
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)

    # 反射成分（R）を分解
    with torch.no_grad():
        model.set_input(input_tensor, targets=None)
        model.forward()
        prediction_R = model.prediction_R.data.cpu().numpy()

    # Rの正規化
    R_min, R_max = prediction_R.min(), prediction_R.max()
    prediction_R = (prediction_R - R_min) / (R_max - R_min) * 255.0
    reflectance_image = prediction_R[0].transpose(1, 2, 0).astype(np.uint8)

    # 反射成分を元のサイズの有効領域にリサイズ
    reflectance_resized = cv2.resize(reflectance_image, (valid_region.shape[1], valid_region.shape[0]))

    # 元のサイズの画像を作成（黒い背景）
    final_reflectance = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    
    # 有効領域の位置を特定
    valid_coords = np.where(valid_mask)
    y_min, y_max = valid_coords[0].min(), valid_coords[0].max()
    x_min, x_max = valid_coords[1].min(), valid_coords[1].max()
    
    # 反射成分を適切な位置に配置
    final_reflectance[y_min:y_max+1, x_min:x_max+1] = reflectance_resized

    # R画像保存
    reflectance_filename = os.path.join(output_dir, f"frame_{frame_num:04d}_reflectance.png")
    cv2.imwrite(reflectance_filename, final_reflectance)
    print(f"Saved Reflectance Image: {reflectance_filename}")

    # 統計データの保存（反射成分）
    reflectance_stats, r_ref, g_ref, b_ref, h_ref, l_ref = calculate_color_stats(reflectance_resized)
    save_color_stats_to_csv(output_dir, frame_num, reflectance_stats, 'reflectance')
    save_h_l_histograms(h_ref, l_ref, output_dir, frame_num, "reflectance")

    return True

def process_image_sequence(image_sequence_dir, output_dir_base, max_frames=None, batch_size=50):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir_base, f"onlyclothing")
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(image_sequence_dir))
    
    if max_frames is not None:
        image_files = image_files[:max_frames]
    
    for batch_start in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[batch_start:batch_start+batch_size]
        
        for frame_num, image_file in enumerate(batch_files, start=batch_start):
            try:
                input_image_path = os.path.join(image_sequence_dir, image_file)
                frame_output_dir = os.path.join(output_dir, f"frame_{frame_num:04d}")
                os.makedirs(frame_output_dir, exist_ok=True)
                
                process_image(input_image_path, frame_output_dir, frame_num)
                
                gc.collect()
                torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                traceback.print_exc()
                continue
    
    print(f"処理が完了しました: {output_dir}")

if __name__ == "__main__":
    image_sequence_dir = ""
    output_dir_base = ""

    process_image_sequence(
        image_sequence_dir,
        output_dir_base,
        max_frames=920,  # 必要に応じて変更
        batch_size=50
    )

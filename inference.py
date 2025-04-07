import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import glob
import os
from sklearn.metrics import jaccard_score

# モデル定義（学習時と同じ設定）
model = smp.Unet(
    encoder_name='vgg11',
    encoder_weights=None,
    classes=3,              # 背景＋横断歩道＋停止線
    activation=None
).cuda()

# 学習済みモデルを読み込み
model.load_state_dict(torch.load('ternaus_usgs.pth'))
model.eval()

# 推論対象フォルダと正解ラベルフォルダ
inference_images = sorted(glob.glob('./dataset/inference/*.png'))
ground_truth_masks = sorted(glob.glob('./dataset/inference_masks/*.png'))
output_dir = './dataset/inference_results/'
os.makedirs(output_dir, exist_ok=True)

# 色付きマスクの色設定
color_map = {
    0: [0, 0, 0],        # 背景（黒）
    1: [42, 125, 209],   # 横断歩道（青）
    2: [36, 179, 83]     # 停止線（緑）
}

# 性能評価のための初期化
scores = []

for image_path, gt_mask_path in zip(inference_images, ground_truth_masks):
    img = cv2.imread(image_path)
    original_size = (img.shape[1], img.shape[0])

    img_resized = cv2.resize(img, (256, 256))
    img_tensor = torch.tensor(img_resized.transpose(2, 0, 1) / 255.).unsqueeze(0).float().cuda()

    with torch.no_grad():
        pred = model(img_tensor)

    pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # 性能評価用の正解マスクの読み込み
    gt_mask_color = cv2.imread(gt_mask_path)
    gt_mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
    for class_id, color in color_map.items():
        gt_mask[np.all(gt_mask_color == color, axis=-1)] = class_id

    # IoU（Intersection over Union）スコア計算
    iou_score = jaccard_score(gt_mask.flatten(), pred_mask_resized.flatten(), average='macro')
    scores.append(iou_score)

    # 可視化用マスク作成
    mask_color = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        mask_color[pred_mask_resized == class_id] = color

    overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)

    base_name = os.path.basename(image_path)
    mask_filename = os.path.join(output_dir, f'mask_{base_name}')
    overlay_filename = os.path.join(output_dir, f'overlay_{base_name}')

    # マスク画像をRGB→BGR変換して保存（表示色の修正）
    mask_color_bgr = cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mask_filename, mask_color_bgr)
    cv2.imwrite(overlay_filename, overlay)

    print(f"推論結果を{mask_filename}と{overlay_filename}に保存しました。IoUスコア：{iou_score:.4f}")

# 性能評価レポートを出力
average_iou = np.mean(scores)
report_path = os.path.join(output_dir, 'performance_report.txt')
with open(report_path, 'w') as f:
    f.write(f'平均IoUスコア: {average_iou:.4f}\n')
    for idx, score in enumerate(scores):
        f.write(f'画像{idx+1} IoUスコア: {score:.4f}\n')

print(f"性能評価レポートを{report_path}に保存しました。平均IoUスコア：{average_iou:.4f}")
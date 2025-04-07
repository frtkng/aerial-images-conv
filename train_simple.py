import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import cv2, glob
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images = sorted(glob.glob(images_path + "/*.png"))
        self.masks = sorted(glob.glob(masks_path + "/*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 入力画像（RGB）を読み込み・リサイズ
        image = cv2.imread(self.images[idx])
        image = cv2.resize(image, (256, 256))
        image = image.transpose(2, 0, 1) / 255.0

        # CVATのカラー画像を読み込み（ラベル画像）
        mask_color = cv2.imread(self.masks[idx])
        mask_color = cv2.resize(mask_color, (256, 256))

        # マスクをクラスごとに作成（2クラス：横断歩道と停止線）
        mask = np.zeros((256, 256), dtype=np.uint8)

        # クラス0: 背景（黒） [0,0,0] → 0（デフォルト）
        # クラス1: 横断歩道（青系）[42,125,209] → 1
        mask[np.all(mask_color == [42,125,209], axis=-1)] = 1

        # クラス2: 停止線（緑系）[36,179,83] → 2
        mask[np.all(mask_color == [36,179,83], axis=-1)] = 2

        # tensor変換（クラスIDでLong型）
        image_tensor = torch.tensor(image).float()
        mask_tensor = torch.tensor(mask).long()

        return image_tensor, mask_tensor

# データセット読み込み
train_dataset = SimpleDataset('./dataset/images', './dataset/masks')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# TernausNetモデル定義 (2クラス＋背景=合計3クラス)
model = smp.Unet(
    encoder_name='vgg11',
    encoder_weights='imagenet',
    classes=3,              # 背景+横断歩道+停止線
    activation=None         # クラス分類時はactivation無し
).cuda()

# 損失関数と最適化関数（多クラス分類用）
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 学習ループ（例：10エポック）
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for img, mask in train_loader:
        img, mask = img.cuda(), mask.cuda()
        optimizer.zero_grad()
        pred = model(img) # 出力は (batch, 3, H, W)
        loss = loss_fn(pred, mask)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss={epoch_loss/len(train_loader)}')

# 学習済みモデルを保存
torch.save(model.state_dict(), 'ternaus_usgs.pth')

import torch
import segmentation_models_pytorch as smp

# TernausNet（VGG11ベースのU-Net）
model = smp.Unet(
    encoder_name='vgg11', 
    encoder_weights='imagenet',  # 事前学習済み
    classes=1,                   # クラス数：まずは停止線・横断歩道をまとめて抽出
    activation='sigmoid'         # sigmoidで二値分類
).cuda()

# GPU動作確認
print("GPU利用可能か:", torch.cuda.is_available())
print("GPU名:", torch.cuda.get_device_name(0))

# ダミーデータで動作確認（1, 3, 256, 256の画像）
x = torch.rand(1, 3, 256, 256).cuda()
y = model(x)
print("出力サイズ:", y.size())

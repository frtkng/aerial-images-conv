# Aerial Images Semantic Segmentation with TernausNet (VGG11 + U-Net)

航空写真（USGS提供）から道路要素（横断歩道・停止線）をセマンティックセグメンテーションするためのプロトタイププロジェクトです。

本リポジトリでは、TernausNet（VGG11エンコーダを使用したU-Net）を用いて道路要素の抽出を試みています。

---

## 🔖 プロジェクト概要

- **入力画像**：USGS航空写真（約30cm/px）
- **対象要素**：交差点付近の横断歩道・停止線
- **モデル**：TernausNet (VGG11エンコーダ使用のU-Net)
- **使用ライブラリ**：[segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)

---

## 📂 フォルダ構成

aerial-images-conv/
├── dataset/
│   ├── train/          # 学習用画像とマスクデータ
│   ├── val/            # 検証用画像とマスクデータ
│   └── inference/      # 推論用画像
├── ternaus_usgs.pth    # 学習済みモデル（TernausNet）
├── train_simple.py     # TernausNetの簡易学習スクリプト
├── inference.py        # TernausNetの推論スクリプト
└── requirements.txt    # Python環境依存関係


---

## 🚀 環境セットアップ手順

### Step 1: Python環境構築 (Ubuntu)

```bash
python3 -m venv ternaus_env
source ternaus_env/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

※事前にCUDA 11.8がインストール済みである必要があります。

###  Step 2: 学習の実行

python train_simple.py

学習が完了すると、ternaus_usgs.pthが生成されます。


### Step 3: 推論の実行
推論用画像をdataset/inferenceフォルダに格納して実行：

python inference.py

推論結果はdataset/inference_resultsに保存されます。

## 🛠️ 使用技術
Python 3.10+

PyTorch（CUDA 11.8）

segmentation-models-pytorch

OpenCV

USGS航空写真データ

## 📌 注意点
現状の精度はプロトタイプ段階のため、学習データ数が少なく限定的です（10枚程度）。

精度向上のためには追加データや解像度の高い航空写真を使用することを推奨します。

## 🚩 今後の改善点
solarisとSpaceNetモデルを用いた航空写真の事前学習済みモデルをベースに再学習を行い、性能を改善する予定です。

新たな改善は別リポジトリで管理します。

## 📖 ライセンス
MIT License
（詳細はLICENSEを参照）

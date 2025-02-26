# clothingalbedoimage
## 概要
このリポジトリには、BIGTIME および ShadingNet を用いたアルベド画像の推定スクリプトが含まれています。

- `decompose.py`: **人物領域** を入力としてアルベド画像を推定し、上半身服領域の色情報を取得します。  
- `decompose_clothingregion.py`: **服領域のみ** を入力としてアルベド画像を推定します。

## 使用方法
それぞれのスクリプトを動作させるには、以下のリポジトリの環境設定に従ってください。

- **BIGTIME**: [unsupervised-learning-intrinsic-images](https://github.com/zhengqili/unsupervised-learning-intrinsic-images)  
- **ShadingNet**: [ShadingNet](https://github.com/Morpheus3000/ShadingNet)

## データセット
- 人物検出領域を **拡大したもの**、**服領域のみ** のデータを **10人分** 用意しています。
- データセット作成用のプログラムも含まれています。
- データセットは以下のリンクからダウンロードできます。
 [データセット (Google Drive)](https://drive.google.com/drive/folders/1SD609c0Cz_uErMlHXnLYC665O2cpEIAh?usp=sharing)

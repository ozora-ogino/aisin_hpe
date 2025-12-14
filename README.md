
# HPE Model Investigation

本コードでは、6DRepNetの公式レポジトリの学習再現環境を提供する。
公式レポジトリ：https://github.com/thohemp/6DRepNet

学習済みログは`output/snapshots`以下に保存されている。本学習にはWANDBを利用しているため、任意で`WANDB_API_KEY`を環境変数に設定してください。

また、オリジナルの学習コードではLossの追跡とベストモデルの判定がなかったため、WANDBの追加とベストモデルのPrintを追加している。また、評価用のコードも新規作成している。（`sixdrepnet/evaluate_epochs.py`)


## 1. データセット準備

※ すでにdatasets/以下に前処理済みのデータを格納済みであるため、こちらはスキップしていただけます。

以下のデータセットをダウンロードして使用します：

* **300W-LP**, **AFLW2000**
  → ダウンロードリンク: [http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)

* **BIWI (Biwi Kinect Head Pose Database)**
  → ダウンロードリンク: [https://icu.ee.ethz.ch/research/datsets.html](https://icu.ee.ethz.ch/research/datsets.html)

ダウンロード後、すべて *datasets* ディレクトリ内に保存します。

---

### **● 300W-LP / AFLW2000 の前処理（ファイルリスト作成）**

次のスクリプトでファイル名リストを作成します：

```sh
python create_filename_list.py --root_dir datasets/300W_LP
```

---

### **● BIWI データセットの前処理（顔の切り出し + 分割）**

BIWI は顔検出器で顔画像を切り出す必要があります。
以下の FSA-Net のスクリプトを利用します：

* 顔切り出し用:
  [https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py](https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py)

* 7:3 の train/val 分割用:
  [https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi_70_30.py](https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi_70_30.py)

切り出す画像サイズは **256×256** に設定します。

---

## 2. 学習

事前学習済みの RepVGG モデル
**"RepVGG-B1g2-train.pth"** を以下からダウンロードします：

[https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq](https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq)

ダウンロードしたファイルは **プロジェクトのルートディレクトリ** に保存します。（こちらについてもDownload済みです）


## 3. トレーニングを実行

```sh
# Docker build
make build

# Start training on Docker (python sixdrepnet/train.py)
make train
```

**CLI arguments**

```py
parser = argparse.ArgumentParser(
	description='Head pose estimation using the 6DRepNet.')
parser.add_argument(
	'--gpu', dest='gpu_id', help='GPU device id to use [0]',
	default=0, type=int)
parser.add_argument(
	'--num_epochs', dest='num_epochs',
	help='Maximum number of training epochs.',
	default=80, type=int)
parser.add_argument(
	'--batch_size', dest='batch_size', help='Batch size.',
	default=80, type=int)
parser.add_argument(
	'--lr', dest='lr', help='Base learning rate.',
	default=0.0001, type=float)
parser.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument(
	'--dataset', dest='dataset', help='Dataset type.',
	default='Pose_300W_LP', type=str) #Pose_300W_LP
parser.add_argument(
	'--data_dir', dest='data_dir', help='Directory path for data.',
	default='datasets/300W_LP', type=str)#BIWI_70_30_train.npz
parser.add_argument(
	'--filename_list', dest='filename_list',
	help='Path to text file containing relative paths for every example.',
	default='datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
parser.add_argument(
	'--output_string', dest='output_string',
	help='String appended to output snapshots.', default='', type=str)
parser.add_argument(
	'--snapshot', dest='snapshot', help='Path of model snapshot.',
	default='', type=str)
```
学習ログは`output/snapshots`以下に保存されます。

## 4. 評価

以下のコマンドで評価が実行され、評価結果は、`output/evaluate`以下に格納されます。

```bash
make evaluate
```

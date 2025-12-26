
# HPE Model Investigation

本コードでは、6DRepNetの公式レポジトリの学習再現環境を提供する。
公式レポジトリ：https://github.com/thohemp/6DRepNet

学習ログは `output/<run_name>/` 以下に保存される。TensorBoard でログを確認可能。

オリジナルの学習コードに以下の機能を追加している：
- TensorBoard によるロギング
- ベストモデルの自動保存
- Mixed Precision (AMP) サポート
- tqdm / Rich によるモダンなコンソール出力
- 評価用スクリプト (`sixdrepnet/evaluate_epochs.py`)


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

```bash
python sixdrepnet/train.py \
  --gpu 0 \
  --num_epochs 30 \
  --batch_size 64 \
  --lr 0.0001 \
  --dataset Pose_300W_LP \
  --data_dir datasets/300W_LP \
  --filename_list datasets/300W_LP/files.txt \
  --val_dataset AFLW2000 \
  --val_data_dir datasets/AFLW2000 \
  --val_filename_list datasets/AFLW2000/files.txt \
  --seed 42 \
  --milestones 10,20 \
  --amp true \
  --save_interval 5
```

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--gpu` | GPU ID | 0 |
| `--num_epochs` | エポック数 | 30 |
| `--batch_size` | バッチサイズ | 64 |
| `--lr` | 学習率 | 0.0001 |
| `--scheduler` | LRスケジューラ有効化 | false |
| `--milestones` | LRスケジューラのマイルストーン | 10,20 |
| `--seed` | 乱数シード | 42 |
| `--amp` | Mixed Precision有効化 | false |
| `--save_interval` | チェックポイント保存間隔 | 1 |
| `--output_dir` | 出力ベースディレクトリ | output |
| `--snapshot` | レジュームするチェックポイント | - |

学習ログは `output/<run_name>/` 以下に保存されます：
```
output/<run_name>/
├── config.json           # 学習設定
├── events.out.tfevents.* # TensorBoardログ
└── checkpoints/
    ├── epoch_1.pt
    ├── epoch_5.pt
    └── best.pt
```

## 4. 評価

以下のコマンドで評価が実行され、評価結果は、`output/evaluate`以下に格納されます。

```bash
make evaluate
```

## 5. 推論・可視化

学習済みモデルを使用して、任意の画像に対してHead Pose推論を行い、結果を可視化します。

```bash
python sixdrepnet/inference.py \
    --snapshot output/<run_name>/checkpoints/best.pt \
    --data_dir path/to/images \
    --output_dir output/inference
```

### オプション

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--snapshot` | 学習済みモデルのパス (必須) | - |
| `--data_dir` | 入力画像のディレクトリ (必須) | - |
| `--output_dir` | 出力ディレクトリ | `output/inference` |
| `--gpu` | GPU ID (-1でCPU使用) | 0 |
| `--img_ext` | 画像ファイルの拡張子 | `.jpg` |
| `--viz_mode` | 可視化モード: `cube` (3Dキューブ) / `axis` (軸矢印) | `cube` |
| `--viz_size` | 可視化オーバーレイのサイズ | 100 |
| `--filename_list` | ファイル名リスト (省略時はディレクトリ内全画像を処理) | None |

### 出力

- 各画像に対してYaw/Pitch/Rollを推論し、3Dポーズを描画した画像を `output_dir` に保存
- `results.json`: 全画像の推論結果をJSON形式で出力

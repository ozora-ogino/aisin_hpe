# Changelog

## [2025-01-XX] - Aisin Data Conversion & AFLW Dataset Fix

### Added
- **Aisin変換スクリプト** (`sixdrepnet/convert_aisin.py`):
  - アイシンデータ（魚眼PNG + モーキャプCSV）を6DRepNet学習用フォーマットに変換
  - InsightFaceによる顔検出・切り出し
  - 出力形式: AFLW（個別jpg+txt）または BIWI（npz）
  - movieID/ImageID マッチング（ゼロパディング対応、int変換で照合）
  - CSVエンコーディング自動検出（cp932, utf-8, shift-jis）
  - 列名またはインデックスによる柔軟な列指定
  - Train/Val分割（movie_id単位 or ランダム）
  - 変換統計・メタデータ出力（meta.json）

  **顔検出アルゴリズム詳細:**

  InsightFace (`insightface.app.FaceAnalysis`) を使用。
  デフォルトモデルパック `buffalo_l` に含まれる **RetinaFace** で顔検出。

  ```
  モデル構成 (buffalo_l):
    - det_10g.onnx: RetinaFace (顔検出、10GFlops)
    - 2d106det.onnx: 2D顔ランドマーク (106点)
    - w600k_r50.onnx: ArcFace (顔認識、ResNet50)
    - genderage.onnx: 性別・年齢推定

  本スクリプトでは det_10g.onnx (RetinaFace) のみ使用。
  ```

  **処理フロー:**
  ```
  Step 1: 顔検出
    faces = detector.get(image_bgr)  # BGR入力必須
    # 複数検出時は最大面積の顔を選択

  Step 2: バウンディングボックス取得
    x1, y1, x2, y2 = face.bbox

  Step 3: マージン追加
    margin = crop_margin * max(width, height)
    # デフォルト: 40% マージン（顔周辺のコンテキスト保持）

  Step 4: 切り出し・リサイズ
    cropped = image[y1-margin:y2+margin, x1-margin:x2+margin]
    resized = cv2.resize(cropped, (img_size, img_size))
  ```

  **顔未検出時の挙動** (`--on_no_face`):
  - `skip`: スキップ（デフォルト、統計に記録）
  - `center`: 画像中心を正方形切り出し
  - `error`: エラー終了

  **オプション:**
  - `--det_size`: 検出解像度（デフォルト: 640）。大きいほど精度↑速度↓
  - `--crop_margin`: マージン比率（デフォルト: 0.4）
  - `--img_size`: 出力画像サイズ（デフォルト: 64）

- **角度補正機能** (`--angle_correction to_camera`):
  - 絶対角度（モーキャプ）からカメラ相対角度への変換
  - 頭部位置（head_x/y/z）とカメラ位置から補正角度を計算
  - 軸設定・符号調整オプション

  **アルゴリズム詳細:**

  モーションキャプチャの角度は「絶対座標系での頭部向き」だが、
  頭部姿勢推定では「カメラから見た相対角度」が必要。

  ```
  入力:
    - yaw_abs, pitch_abs: 絶対角度（モーキャプ、度単位）
    - head_xyz: 頭部位置 [x, y, z] (mm)
    - cam_xyz: カメラ位置 [x, y, z] (mm)

  Step 1: 頭部→カメラへのベクトル計算
    v = cam_xyz - head_xyz

  Step 2: ベクトルを各軸成分に分解
    vf = v[forward_axis]  # 前方成分
    vr = v[right_axis]    # 右方成分
    vu = v[up_axis]       # 上方成分

  Step 3: カメラ方向のyaw/pitchを計算
    yaw_to_cam = atan2(vr, vf)  # 水平面での角度
    pitch_to_cam = atan2(vu, sqrt(vf² + vr²))  # 仰角

  Step 4: 相対角度を計算
    yaw_corr = wrap180(yaw_abs - yaw_to_cam)
    pitch_corr = clip(pitch_abs - pitch_to_cam, -90, 90)

  出力:
    - yaw_corr: カメラを正面(0°)とした相対yaw
    - pitch_corr: カメラを正面(0°)とした相対pitch
  ```

  **補正の意味:**
  - 被験者がカメラを見ている時 → yaw_corr ≈ 0, pitch_corr ≈ 0
  - 被験者がカメラより右を見ている → yaw_corr > 0
  - 被験者がカメラより上を見ている → pitch_corr > 0

  **座標系設定オプション:**
  - `--forward_axis`: 前方軸（デフォルト: x）
  - `--right_axis`: 右軸（デフォルト: z）
  - `--up_axis`: 上軸（デフォルト: y）
  - `--yaw_sign`, `--pitch_sign`: 符号反転（-1.0で反転）

- **Aisinデータセットエイリアス**: `datasets.py`の`getDataset()`に'Aisin'追加（BIWI形式用）

- **依存パッケージ**:
  - `insightface >= 0.7.0`
  - `onnxruntime-gpu >= 1.15.0`

### Changed
- **AFLWデータセットクラス** (`datasets.py`):
  - ビン分類ラベルから回転行列(3x3)を返すように変更
  - 6DRepNetのGeodesicLoss学習に対応
  - `roll *= -1` の補正を削除（Aisinデータには不要）

### Fixed
- **BGR/RGB色順序**: InsightFaceはBGR入力を期待するため、検出後にRGB変換
- **ImageIDエラーハンドリング**: `pd.to_numeric(..., errors='coerce')` → `dropna()` → `astype(int)` の順序で安全に変換

### Usage
```bash
# 変換
python sixdrepnet/convert_aisin.py \
  --aisin_root datasets/aisin \
  --output_dir datasets/aisin/converted \
  --angle_correction to_camera

# 学習（AFLW形式）
python sixdrepnet/train.py \
  --dataset AFLW \
  --data_dir datasets/aisin/converted \
  --filename_list datasets/aisin/converted/files.txt
```

---

## [2025-01-06] - Training Script Modernization

### Added
- **Mixed Precision Training (AMP)**: `--amp true` で有効化、学習高速化・メモリ削減
- **再現性サポート**: `--seed` 引数で乱数シードを設定可能
- **モダンなコンソール出力**:
  - tqdm によるプログレスバー (loss, GPU メモリ表示)
  - Rich による設定テーブル、エポックサマリー、学習完了サマリー
- **config.json**: 学習設定を JSON 形式で自動保存
- **新しい引数**:
  - `--seed`: 乱数シード (default: 42)
  - `--milestones`: LR スケジューラのマイルストーン (default: "10,20")
  - `--amp`: Mixed Precision 有効化 (default: false)
  - `--save_interval`: チェックポイント保存間隔 (default: 1)
  - `--output_dir`: 出力ベースディレクトリ (default: "output")
- **TensorBoard ログ追加**:
  - `train/epoch_time`: エポック毎の学習時間
  - `train/gpu_memory_mb`: GPU メモリ使用量

### Changed
- **ディレクトリ構造を統一**:
  ```
  # Before
  output/logs/<run_name>/        # TensorBoard
  output/snapshots/<run_name>/   # Checkpoints

  # After
  output/<run_name>/
  ├── config.json
  ├── events.out.tfevents.*
  └── checkpoints/
      ├── epoch_1.pt
      └── best.pt
  ```
- **チェックポイント拡張子**: `.tar` → `.pt`
- **チェックポイント命名**: `_epoch_1.tar` → `epoch_1.pt`, `_best_model.tar` → `best.pt`
- **DataLoader**: `pin_memory=True`, `non_blocking=True` で GPU 転送を高速化
- **cuDNN**: `cudnn.benchmark=True` で自動チューニング有効化
- **torch.load**: `map_location`, `weights_only=True` を追加

### Removed
- **未使用のインポート**: `math`, `re`, `cv2`, `torch.nn.functional`, `model_zoo`, `torchvision`, `matplotlib.pyplot`, `PIL.Image`, `SixDRepNet2`
- **不要な引数**: `--log_dir`, `--output_string`
- **冗長なコード**: `torch.Tensor(images)` の不要な変換、`epoch % 1 == 0` の常に true な条件

### Fixed
- **変数名**: `iter` → `num_iters` (Python 組み込み関数の上書きを回避)

### Dependencies
- `tqdm >= 4.62.0` を追加
- `rich >= 13.0.0` を追加

### Documentation
- `README.md`: 新しいディレクトリ構造、引数、機能を反映
- `CLAUDE.md`: 同様に更新

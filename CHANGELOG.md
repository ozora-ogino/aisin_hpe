# Changelog

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

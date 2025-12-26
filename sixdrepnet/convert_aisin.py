#!/usr/bin/env python3
"""
Convert Aisin head pose data (fisheye PNG + motion capture CSV) to 6DRepNet format.

This script converts Aisin-provided data to AFLW or BIWI format for training.

Usage:
    # AFLW format (default) - individual images + .txt annotations
    python sixdrepnet/convert_aisin.py \
        --aisin_root /path/to/AisinData \
        --output_dir datasets/Aisin/output

    # BIWI format (.npz)
    python sixdrepnet/convert_aisin.py \
        --aisin_root /path/to/AisinData \
        --output_dir datasets/Aisin/output \
        --output_format npz
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# InsightFace for face detection
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Face detection will be disabled.")


@dataclass
class ConversionStats:
    """Track statistics during conversion."""
    total_movies: int = 0
    total_csv_rows: int = 0
    total_images_found: int = 0
    matched_pairs: int = 0
    unmatched_csv_rows: int = 0
    unmatched_images: int = 0
    missing_csv: int = 0
    face_detection_success: int = 0
    face_detection_failed: int = 0
    processing_errors: int = 0
    train_samples: int = 0
    val_samples: int = 0

    def report(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 50,
            "Conversion Statistics",
            "=" * 50,
            f"Total movies processed: {self.total_movies}",
            f"Total images found: {self.total_images_found}",
            f"Total CSV rows: {self.total_csv_rows}",
            f"Matched image-label pairs: {self.matched_pairs}",
            f"Unmatched images (no CSV row): {self.unmatched_images}",
            f"Unmatched CSV rows (no image): {self.unmatched_csv_rows}",
            f"Missing CSV files: {self.missing_csv}",
            f"Face detection success: {self.face_detection_success}",
            f"Face detection failed: {self.face_detection_failed}",
            f"Processing errors: {self.processing_errors}",
            "-" * 50,
            f"Train samples: {self.train_samples}",
            f"Val samples: {self.val_samples}",
            "=" * 50,
        ]
        return "\n".join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert Aisin head pose data to 6DRepNet format (BIWI-compatible .npz)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output paths
    parser.add_argument('--aisin_root', type=str, required=True,
                        help='Root directory of Aisin data (contains image/ and motion/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for converted dataset')

    # CSV column configuration
    parser.add_argument('--yaw_col', type=str, default='頭方位角',
                        help='Column name or 0-based index for yaw (degrees)')
    parser.add_argument('--pitch_col', type=str, default='頭仰角',
                        help='Column name or 0-based index for pitch (degrees)')
    parser.add_argument('--image_id_col', type=str, default='ImageID',
                        help='Column name or 0-based index for ImageID')
    parser.add_argument('--csv_skip_rows', type=int, default=11,
                        help='Number of metadata rows to skip in CSV (header is on row 11)')

    # Output format
    parser.add_argument('--output_format', type=str, default='aflw',
                        choices=['npz', 'aflw', 'both'],
                        help='Output format: aflw (individual images + txt), npz (BIWI), or both')

    # Image processing / Face cropping
    parser.add_argument('--img_size', type=int, default=64,
                        help='Output image size (square)')
    parser.add_argument('--crop_margin', type=float, default=0.4,
                        help='Margin ratio around detected face')
    parser.add_argument('--on_no_face', type=str, default='skip',
                        choices=['skip', 'center', 'error'],
                        help='Behavior when no face detected')
    parser.add_argument('--det_size', type=int, default=640,
                        help='InsightFace detection size')

    # Angle correction
    parser.add_argument('--angle_correction', type=str, default='none',
                        choices=['none', 'to_camera'],
                        help='Angle correction: none or to_camera (relative to camera)')
    parser.add_argument('--head_x_col', type=str, default='頭x',
                        help='Column name for head X position')
    parser.add_argument('--head_y_col', type=str, default='頭y',
                        help='Column name for head Y position')
    parser.add_argument('--head_z_col', type=str, default='頭z',
                        help='Column name for head Z position')
    parser.add_argument('--camera_x', type=float, default=0.0,
                        help='Camera X position (mm)')
    parser.add_argument('--camera_y', type=float, default=1576.0,
                        help='Camera Y position (mm)')
    parser.add_argument('--camera_z', type=float, default=-630.0,
                        help='Camera Z position (mm)')
    parser.add_argument('--forward_axis', type=str, default='x',
                        choices=['x', 'y', 'z'], help='Forward axis')
    parser.add_argument('--right_axis', type=str, default='z',
                        choices=['x', 'y', 'z'], help='Right axis')
    parser.add_argument('--up_axis', type=str, default='y',
                        choices=['x', 'y', 'z'], help='Up axis')
    parser.add_argument('--yaw_sign', type=float, default=1.0,
                        help='Sign multiplier for yaw (1.0 or -1.0)')
    parser.add_argument('--pitch_sign', type=float, default=1.0,
                        help='Sign multiplier for pitch (1.0 or -1.0)')

    # Data splitting
    parser.add_argument('--val_split', type=float, default=0.0,
                        help='Fraction for validation split (0.0 to 1.0)')
    parser.add_argument('--split_by', type=str, default='movie_id',
                        choices=['movie_id', 'random'],
                        help='How to split train/val')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_image_filename(filename: str) -> Optional[Tuple[str, str, str, str, int]]:
    """
    Parse Aisin image filename.

    Format: {movieID}_{yyMMDD}_{hhmmss}_{camID}_{ImageID}.png
    Example: 01_250116_111927_0_00193.png

    Returns: (movie_id, date, time, cam_id, image_id as int) or None if parse fails
    """
    # Remove extension
    name = Path(filename).stem

    # Split by underscore
    parts = name.split('_')
    if len(parts) < 5:
        return None

    try:
        movie_id = parts[0]
        date = parts[1]
        time = parts[2]
        cam_id = parts[3]
        # ImageID is the last part, convert to int to handle zero-padding
        image_id = int(parts[4])
        return (movie_id, date, time, cam_id, image_id)
    except (ValueError, IndexError):
        return None


def parse_folder_name(folder_name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse Aisin image folder name.

    Format: {movieID}_{yyMMDD}_{hhmmss}_{camID}
    Example: 01_250116_111927_0

    Returns: (movie_id, date, time, cam_id) or None if parse fails
    """
    parts = folder_name.split('_')
    if len(parts) < 4:
        return None

    try:
        movie_id = parts[0]
        date = parts[1]
        time = parts[2]
        cam_id = parts[3]
        return (movie_id, date, time, cam_id)
    except (ValueError, IndexError):
        return None


def parse_csv_movie_id(csv_filename: str) -> Optional[str]:
    """
    Parse movie ID from CSV filename.

    Format: {movieID}_解析結果ファイル.csv
    Example: 001_解析結果ファイル.csv -> "001"

    Returns: movie_id string or None
    """
    name = Path(csv_filename).stem
    parts = name.split('_')
    if len(parts) >= 1:
        return parts[0]
    return None


def discover_aisin_data(aisin_root: str) -> Dict[int, Dict]:
    """
    Scan AisinData directory and build mapping.

    Returns dict keyed by movie_id (as int) with structure:
    {
        movie_id: {
            'csv_path': str or None,
            'image_folders': [
                {
                    'folder_path': str,
                    'folder_name': str,
                    'cam_id': str,
                    'images': {image_id (int): full_path, ...}
                }
            ]
        }
    }
    """
    aisin_root = Path(aisin_root)
    image_dir = aisin_root / 'image'
    motion_dir = aisin_root / 'motion'

    data = {}

    # Scan image folders
    if image_dir.exists():
        for folder in sorted(image_dir.iterdir()):
            if not folder.is_dir():
                continue

            parsed = parse_folder_name(folder.name)
            if parsed is None:
                logging.warning(f"Could not parse folder name: {folder.name}")
                continue

            movie_id_str, _, _, cam_id = parsed
            try:
                movie_id = int(movie_id_str)
            except ValueError:
                logging.warning(f"Invalid movie_id in folder: {folder.name}")
                continue

            if movie_id not in data:
                data[movie_id] = {'csv_path': None, 'image_folders': []}

            # Scan images in folder
            images = {}
            for img_file in folder.glob('*.png'):
                parsed_img = parse_image_filename(img_file.name)
                if parsed_img:
                    _, _, _, _, image_id = parsed_img
                    images[image_id] = str(img_file)

            data[movie_id]['image_folders'].append({
                'folder_path': str(folder),
                'folder_name': folder.name,
                'cam_id': cam_id,
                'images': images
            })

    # Scan CSV files
    if motion_dir.exists():
        for csv_file in sorted(motion_dir.glob('*.csv')):
            movie_id_str = parse_csv_movie_id(csv_file.name)
            if movie_id_str is None:
                logging.warning(f"Could not parse CSV filename: {csv_file.name}")
                continue

            try:
                movie_id = int(movie_id_str)
            except ValueError:
                logging.warning(f"Invalid movie_id in CSV: {csv_file.name}")
                continue

            if movie_id not in data:
                data[movie_id] = {'csv_path': None, 'image_folders': []}

            data[movie_id]['csv_path'] = str(csv_file)

    return data


def resolve_column(df: pd.DataFrame, col_spec: str) -> str:
    """
    Resolve column specification to actual column name.

    If col_spec is a numeric string, treat as 0-based index.
    Otherwise, treat as column name.
    """
    # Check if it's a numeric index
    if col_spec.isdigit():
        idx = int(col_spec)
        if idx >= len(df.columns):
            raise KeyError(f"Column index {idx} out of range. Available columns: {list(df.columns)}")
        return df.columns[idx]
    else:
        if col_spec not in df.columns:
            raise KeyError(f"Column '{col_spec}' not found. Available columns: {list(df.columns)}")
        return col_spec


def load_motion_csv(csv_path: str, skip_rows: int,
                    yaw_col: str, pitch_col: str, image_id_col: str,
                    head_x_col: str = None, head_y_col: str = None,
                    head_z_col: str = None) -> pd.DataFrame:
    """
    Load motion capture CSV with proper encoding handling.

    Returns DataFrame with columns: ['image_id', 'yaw', 'pitch']
    and optionally ['head_x', 'head_y', 'head_z'] for angle correction.
    """
    # Try multiple encodings (cp932 first for Japanese CSV)
    encodings = ['cp932', 'utf-8', 'utf-8-sig', 'shift-jis', 'euc-jp']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, skiprows=skip_rows, encoding=encoding)
            logging.debug(f"Successfully loaded {csv_path} with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.warning(f"Error loading {csv_path} with {encoding}: {e}")
            continue

    if df is None:
        raise ValueError(f"Could not decode CSV with any encoding: {csv_path}")

    # Resolve column names
    try:
        yaw_col_name = resolve_column(df, yaw_col)
        pitch_col_name = resolve_column(df, pitch_col)
        image_id_col_name = resolve_column(df, image_id_col)
    except KeyError as e:
        raise ValueError(f"Column resolution failed for {csv_path}: {e}")

    # Extract columns with proper error handling
    result_data = {
        'image_id': pd.to_numeric(df[image_id_col_name], errors='coerce'),
        'yaw': pd.to_numeric(df[yaw_col_name], errors='coerce'),
        'pitch': pd.to_numeric(df[pitch_col_name], errors='coerce')
    }

    # Add head position columns if requested (for angle correction)
    if head_x_col and head_y_col and head_z_col:
        try:
            head_x_col_name = resolve_column(df, head_x_col)
            head_y_col_name = resolve_column(df, head_y_col)
            head_z_col_name = resolve_column(df, head_z_col)
            result_data['head_x'] = pd.to_numeric(df[head_x_col_name], errors='coerce')
            result_data['head_y'] = pd.to_numeric(df[head_y_col_name], errors='coerce')
            result_data['head_z'] = pd.to_numeric(df[head_z_col_name], errors='coerce')
        except KeyError as e:
            logging.warning(f"Head position columns not found in {csv_path}: {e}")

    result = pd.DataFrame(result_data)

    # Remove rows with NaN in essential columns
    result = result.dropna(subset=['image_id', 'yaw', 'pitch'])

    # Convert image_id to int after dropna
    result['image_id'] = result['image_id'].astype(int)

    return result


def init_face_detector(det_size: int = 640) -> Optional[Any]:
    """Initialize InsightFace face detector."""
    if not INSIGHTFACE_AVAILABLE:
        logging.error("InsightFace is not available. Please install it: pip install insightface onnxruntime-gpu")
        return None

    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(det_size, det_size))
        logging.info(f"InsightFace initialized with det_size={det_size}")
        return app
    except Exception as e:
        logging.error(f"Failed to initialize InsightFace: {e}")
        return None


def detect_and_crop_face(image: np.ndarray, detector,
                         crop_margin: float = 0.4,
                         target_size: int = 64) -> Optional[np.ndarray]:
    """
    Detect face and crop with margin.

    Returns cropped and resized face image, or None if no face detected.
    """
    if detector is None:
        return None

    try:
        faces = detector.get(image)

        if len(faces) == 0:
            return None

        # Select largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Calculate margin
        w, h = x2 - x1, y2 - y1
        margin = int(crop_margin * max(w, h))

        # Expand bounding box with margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)

        # Crop
        cropped = image[y1:y2, x1:x2]

        # Resize to target size
        resized = cv2.resize(cropped, (target_size, target_size))

        return resized

    except Exception as e:
        logging.debug(f"Face detection error: {e}")
        return None


def crop_center(image: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Crop center region of image and resize."""
    h, w = image.shape[:2]

    # Determine crop size (square, centered)
    crop_size = min(h, w)
    y1 = (h - crop_size) // 2
    x1 = (w - crop_size) // 2

    cropped = image[y1:y1+crop_size, x1:x1+crop_size]
    resized = cv2.resize(cropped, (target_size, target_size))

    return resized


def wrap180(deg: float) -> float:
    """Wrap angle to [-180, 180] range."""
    return (deg + 180.0) % 360.0 - 180.0


def get_axis_component(v: np.ndarray, axis: str) -> float:
    """Get component of vector along specified axis."""
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    return float(v[axis_map[axis]])


def correct_angle_to_camera(yaw_abs: float, pitch_abs: float,
                            head_xyz: np.ndarray, cam_xyz: np.ndarray,
                            forward_axis: str = 'x',
                            right_axis: str = 'z',
                            up_axis: str = 'y') -> Tuple[float, float]:
    """
    Convert absolute yaw/pitch to camera-relative angles.

    The idea: when looking at the camera, yaw_corr should be ~0.
    We compute the direction from head to camera, extract its yaw/pitch,
    and subtract from the absolute angles.

    Args:
        yaw_abs: Absolute yaw in degrees
        pitch_abs: Absolute pitch in degrees
        head_xyz: Head position [x, y, z] in mm
        cam_xyz: Camera position [x, y, z] in mm
        forward_axis: Which axis is forward ('x', 'y', or 'z')
        right_axis: Which axis is right ('x', 'y', or 'z')
        up_axis: Which axis is up ('x', 'y', or 'z')

    Returns:
        (yaw_corrected, pitch_corrected) in degrees
    """
    # Vector from head to camera
    v = cam_xyz - head_xyz

    # Get components along each axis
    vf = get_axis_component(v, forward_axis)  # forward
    vr = get_axis_component(v, right_axis)    # right
    vu = get_axis_component(v, up_axis)       # up

    # Compute yaw/pitch to camera
    yaw_to_cam = np.degrees(np.arctan2(vr, vf))
    pitch_to_cam = np.degrees(np.arctan2(vu, np.sqrt(vf * vf + vr * vr)))

    # Subtract to get camera-relative angles
    yaw_corr = wrap180(yaw_abs - yaw_to_cam)
    pitch_corr = pitch_abs - pitch_to_cam
    pitch_corr = float(np.clip(pitch_corr, -90.0, 90.0))

    return yaw_corr, pitch_corr


def process_images(data: Dict[int, Dict],
                   detector,
                   yaw_col: str, pitch_col: str, image_id_col: str,
                   csv_skip_rows: int,
                   img_size: int,
                   crop_margin: float,
                   on_no_face: str,
                   stats: ConversionStats,
                   # Angle correction parameters
                   angle_correction: str = 'none',
                   head_x_col: str = '頭x',
                   head_y_col: str = '頭y',
                   head_z_col: str = '頭z',
                   camera_x: float = 0.0,
                   camera_y: float = 1576.0,
                   camera_z: float = -630.0,
                   forward_axis: str = 'x',
                   right_axis: str = 'z',
                   up_axis: str = 'y',
                   yaw_sign: float = 1.0,
                   pitch_sign: float = 1.0) -> List[Dict]:
    """
    Process all images and match with CSV labels.

    Returns list of processed samples:
    [{'image': np.ndarray, 'yaw': float, 'pitch': float, 'roll': float, 'movie_id': int}, ...]
    """
    samples = []

    for movie_id, movie_data in tqdm(sorted(data.items()), desc="Processing movies"):
        stats.total_movies += 1

        csv_path = movie_data['csv_path']
        if csv_path is None:
            logging.warning(f"No CSV file for movie_id: {movie_id}")
            stats.missing_csv += 1
            continue

        # Load CSV (include head position columns if angle correction enabled)
        try:
            if angle_correction == 'to_camera':
                csv_df = load_motion_csv(
                    csv_path, csv_skip_rows, yaw_col, pitch_col, image_id_col,
                    head_x_col=head_x_col, head_y_col=head_y_col, head_z_col=head_z_col
                )
            else:
                csv_df = load_motion_csv(csv_path, csv_skip_rows, yaw_col, pitch_col, image_id_col)
            stats.total_csv_rows += len(csv_df)
        except Exception as e:
            logging.error(f"Failed to load CSV for movie_id {movie_id}: {e}")
            stats.missing_csv += 1
            continue

        # Build lookup by image_id
        csv_lookup = {int(row['image_id']): row for _, row in csv_df.iterrows()}
        csv_matched = set()

        # Process each image folder
        for folder_info in movie_data['image_folders']:
            for image_id, image_path in folder_info['images'].items():
                stats.total_images_found += 1

                # Check if we have a matching CSV row
                if image_id not in csv_lookup:
                    stats.unmatched_images += 1
                    continue

                csv_row = csv_lookup[image_id]
                csv_matched.add(image_id)

                # Load image (keep BGR for InsightFace)
                try:
                    image_bgr = cv2.imread(image_path)
                    if image_bgr is None:
                        logging.warning(f"Failed to load image: {image_path}")
                        stats.processing_errors += 1
                        continue
                except Exception as e:
                    logging.warning(f"Error loading image {image_path}: {e}")
                    stats.processing_errors += 1
                    continue

                # Face detection and cropping (InsightFace expects BGR)
                cropped_bgr = detect_and_crop_face(image_bgr, detector, crop_margin, img_size)

                if cropped_bgr is None:
                    stats.face_detection_failed += 1

                    if on_no_face == 'skip':
                        continue
                    elif on_no_face == 'center':
                        cropped_bgr = crop_center(image_bgr, img_size)
                    elif on_no_face == 'error':
                        raise RuntimeError(f"No face detected in {image_path}")
                else:
                    stats.face_detection_success += 1

                # Convert BGR to RGB for output
                cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

                # Get angles with sign adjustment
                yaw = float(csv_row['yaw']) * yaw_sign
                pitch = float(csv_row['pitch']) * pitch_sign

                # Apply angle correction if enabled
                if angle_correction == 'to_camera' and 'head_x' in csv_row:
                    head_xyz = np.array([
                        float(csv_row['head_x']),
                        float(csv_row['head_y']),
                        float(csv_row['head_z'])
                    ])
                    cam_xyz = np.array([camera_x, camera_y, camera_z])
                    yaw, pitch = correct_angle_to_camera(
                        yaw, pitch, head_xyz, cam_xyz,
                        forward_axis, right_axis, up_axis
                    )

                # Add sample
                samples.append({
                    'image': cropped_rgb,
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': 0.0,  # Roll is always 0 for Aisin data
                    'movie_id': movie_id
                })
                stats.matched_pairs += 1

        # Count unmatched CSV rows
        stats.unmatched_csv_rows += len(csv_lookup) - len(csv_matched)

    return samples


def split_samples(samples: List[Dict],
                  val_split: float,
                  split_by: str,
                  seed: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into train and validation sets.

    If split_by='movie_id', splits by movie to prevent data leakage.
    If split_by='random', random split.
    """
    if val_split <= 0:
        return samples, []

    np.random.seed(seed)

    if split_by == 'movie_id':
        # Get unique movie IDs
        movie_ids = list(set(s['movie_id'] for s in samples))
        np.random.shuffle(movie_ids)

        n_val = max(1, int(len(movie_ids) * val_split))
        val_movie_ids = set(movie_ids[:n_val])

        train_samples = [s for s in samples if s['movie_id'] not in val_movie_ids]
        val_samples = [s for s in samples if s['movie_id'] in val_movie_ids]

    else:  # random
        indices = np.arange(len(samples))
        np.random.shuffle(indices)

        n_val = int(len(samples) * val_split)
        val_indices = set(indices[:n_val])

        train_samples = [s for i, s in enumerate(samples) if i not in val_indices]
        val_samples = [s for i, s in enumerate(samples) if i in val_indices]

    return train_samples, val_samples


def save_npz(samples: List[Dict], output_path: str):
    """
    Save samples to BIWI-compatible .npz format.

    Format:
    - image: (N, H, W, 3) uint8 array
    - pose: (N, 3) float array of [yaw, pitch, roll] in degrees
    """
    if len(samples) == 0:
        logging.warning(f"No samples to save to {output_path}")
        return

    images = np.array([s['image'] for s in samples], dtype=np.uint8)
    poses = np.array([[s['yaw'], s['pitch'], s['roll']] for s in samples], dtype=np.float32)

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, image=images, pose=poses)
    logging.info(f"Saved {len(samples)} samples to {output_path}")


def save_aflw_format(samples: List[Dict], output_dir: str, split_name: str = ""):
    """
    Save samples in AFLW-like format (individual image files + .txt annotations).

    Structure:
    output_dir/
      images/
        image_00001.jpg
        image_00002.jpg
        ...
      image_00001.txt  (annotation: index yaw_rad pitch_rad roll_rad)
      image_00002.txt
      ...
      files.txt  (list of relative paths without extension)

    Angles are stored in RADIANS to match AFLW format.
    """
    if len(samples) == 0:
        logging.warning(f"No samples to save")
        return

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    file_list = []

    for i, sample in enumerate(tqdm(samples, desc=f"Saving {split_name} images")):
        # Generate filename
        img_name = f"image_{i:05d}"
        img_path = images_dir / f"{img_name}.jpg"
        txt_path = images_dir / f"{img_name}.txt"  # Save txt in same dir as images

        # Save image (convert RGB to BGR for OpenCV)
        img_bgr = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), img_bgr)

        # Convert degrees to radians for AFLW format
        yaw_rad = sample['yaw'] * np.pi / 180.0
        pitch_rad = sample['pitch'] * np.pi / 180.0
        roll_rad = sample['roll'] * np.pi / 180.0

        # Save annotation (AFLW format: index yaw pitch roll)
        with open(txt_path, 'w') as f:
            f.write(f"0 {yaw_rad:.6f} {pitch_rad:.6f} {roll_rad:.6f}\n")

        # Add to file list (relative path without extension)
        file_list.append(f"images/{img_name}")

    # Save files.txt
    suffix = f"_{split_name}" if split_name else ""
    files_txt_path = output_path / f"files{suffix}.txt"
    with open(files_txt_path, 'w') as f:
        f.write('\n'.join(file_list))

    logging.info(f"Saved {len(samples)} images to {images_dir}")
    logging.info(f"Saved file list to {files_txt_path}")


def save_metadata(output_dir: str, args, stats: ConversionStats, column_mapping: Dict):
    """Save conversion metadata to JSON."""
    meta = {
        'timestamp': datetime.now().isoformat(),
        'arguments': vars(args),
        'seed': args.seed,
        'column_mapping': column_mapping,
        'statistics': asdict(stats)
    }

    meta_path = Path(output_dir) / 'meta.json'
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved metadata to {meta_path}")


def main():
    args = parse_args()
    setup_logging(args.verbose)

    logging.info("=" * 50)
    logging.info("Aisin Data Conversion Script")
    logging.info("=" * 50)

    # Validate input directory
    aisin_root = Path(args.aisin_root)
    if not aisin_root.exists():
        logging.error(f"Aisin root directory not found: {aisin_root}")
        sys.exit(1)

    # Initialize face detector
    detector = init_face_detector(args.det_size)
    if detector is None:
        logging.error("Failed to initialize face detector. Exiting.")
        sys.exit(1)

    # Initialize statistics
    stats = ConversionStats()

    # Discover data
    logging.info(f"Scanning Aisin data directory: {aisin_root}")
    data = discover_aisin_data(args.aisin_root)
    logging.info(f"Found {len(data)} movies")

    # Process images
    logging.info("Processing images...")
    if args.angle_correction != 'none':
        logging.info(f"Angle correction enabled: {args.angle_correction}")
        logging.info(f"Camera position: ({args.camera_x}, {args.camera_y}, {args.camera_z})")

    samples = process_images(
        data=data,
        detector=detector,
        yaw_col=args.yaw_col,
        pitch_col=args.pitch_col,
        image_id_col=args.image_id_col,
        csv_skip_rows=args.csv_skip_rows,
        img_size=args.img_size,
        crop_margin=args.crop_margin,
        on_no_face=args.on_no_face,
        stats=stats,
        # Angle correction parameters
        angle_correction=args.angle_correction,
        head_x_col=args.head_x_col,
        head_y_col=args.head_y_col,
        head_z_col=args.head_z_col,
        camera_x=args.camera_x,
        camera_y=args.camera_y,
        camera_z=args.camera_z,
        forward_axis=args.forward_axis,
        right_axis=args.right_axis,
        up_axis=args.up_axis,
        yaw_sign=args.yaw_sign,
        pitch_sign=args.pitch_sign
    )

    logging.info(f"Total samples processed: {len(samples)}")

    if len(samples) == 0:
        logging.error("No samples were processed. Check your data and settings.")
        sys.exit(1)

    # Split train/val
    train_samples, val_samples = split_samples(
        samples=samples,
        val_split=args.val_split,
        split_by=args.split_by,
        seed=args.seed
    )

    stats.train_samples = len(train_samples)
    stats.val_samples = len(val_samples)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.val_split > 0:
        # Save train and val separately
        if args.output_format in ['npz', 'both']:
            train_path = output_dir / "aisin_train.npz"
            val_path = output_dir / "aisin_val.npz"
            save_npz(train_samples, str(train_path))
            save_npz(val_samples, str(val_path))

        if args.output_format in ['aflw', 'both']:
            train_dir = output_dir / "train"
            val_dir = output_dir / "val"
            save_aflw_format(train_samples, str(train_dir), "train")
            save_aflw_format(val_samples, str(val_dir), "val")
    else:
        # Save all as training data
        if args.output_format in ['npz', 'both']:
            save_npz(train_samples, str(output_dir / "aisin.npz"))

        if args.output_format in ['aflw', 'both']:
            save_aflw_format(train_samples, str(output_dir), "")

    # Save metadata
    column_mapping = {
        'yaw': args.yaw_col,
        'pitch': args.pitch_col,
        'image_id': args.image_id_col
    }
    save_metadata(str(output_dir), args, stats, column_mapping)

    # Print statistics
    print(stats.report())

    logging.info("Conversion complete!")


if __name__ == '__main__':
    main()

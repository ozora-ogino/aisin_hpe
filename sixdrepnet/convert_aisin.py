#!/usr/bin/env python3
"""
Convert Aisin head pose data (fisheye PNG + motion capture CSV) to 6DRepNet format.

- AFLW / BIWI compatible output
- InsightFace face crop

============================================================
Hard-coded correction (matches your successful vis settings)
============================================================

We use:
  - Head azimuth (SKYCOM) + hard-coded conversion
  - Head position (head_x, head_y, head_z) from CSV for LOS correction using xz plane
  - Final yaw is relative to "head -> camera" direction (camera direction = 0deg)

Final formula (degrees):
  yaw_base = wrap180( YAW_SIGN * az_to_yaw_zref(azimuth_deg) + YAW_OFFSET_DEG )

  beta_c2p = atan2( (head_x - cam_x), (head_z - cam_z) )         # camera -> head
  yaw_p2c  = wrap180( LOS_YAW_SIGN * (beta_c2p + 180) + LOS_YAW_OFFSET_DEG )  # head -> camera

  yaw_out  = wrap180( yaw_base - yaw_p2c )

Pitch:
  pitch_out = wrap180( PITCH_SIGN * pitch_deg + PITCH_OFFSET_DEG )

NOTE:
- Correction parameters are HARD-CODED (no argparse) as requested.
- Head xyz columns are auto-detected (best effort). If detection fails, you can edit
  HEAD_X_COL/HED_Y_COL/HEAD_Z_COL below (hardcode names).
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# InsightFace
# ============================================================
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Face detection will be disabled.")


# ============================================================
# HARD-CODED correction parameters (validated by your command)
# ============================================================

# --- camera origin in the same mocap coordinate (mm)
CAM_X_MM = 0.0
CAM_Y_MM = 0.0
CAM_Z_MM = 0.0

# --- LOS plane axes (you said xz is correct)
LOS_AXES = "xz"  # right=x, forward=z

# --- SKYCOM azimuth interpretation (matches: --head_azimuth_mode x0_ccw_to_z)
HEAD_AZIMUTH_MODE = "x0_ccw_to_z"

# --- Base yaw mapping (matches: --yaw_sign -1 --yaw_offset_deg 90)
YAW_SIGN = -1.0
YAW_OFFSET_DEG = 90.0

# --- LOS yaw mapping (matches: --los_yaw_sign -1 --los_yaw_offset_deg 0)
LOS_YAW_SIGN = -1.0
LOS_YAW_OFFSET_DEG = 0.0

# --- Pitch mapping (keep simple; adjust later if needed)
PITCH_SIGN = 1.0
PITCH_OFFSET_DEG = 0.0

# --- If you want to hardcode head xyz column names, set them here (None = auto-detect)
HEAD_X_COL: Optional[str] = None
HEAD_Y_COL: Optional[str] = None
HEAD_Z_COL: Optional[str] = None

# ============================================================
# Stats
# ============================================================
@dataclass
class ConversionStats:
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
        return "\n".join([
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
            "=" * 50,
        ])


# ============================================================
# Utils
# ============================================================
def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    return s.replace(" ", "").replace("　", "").strip().lower()


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cand = [norm_text(c) for c in candidates]
    for c in df.columns:
        nc = norm_text(c)
        if nc in cand:
            return c
        for cc in cand:
            if cc and cc in nc:
                return c
    return None


def normalize_col_for_suffix(col: str) -> str:
    s = norm_text(col)
    s = re.sub(r"\[.*\]$", "", s)
    s = re.sub(r"\(.*\)$", "", s)
    return s


def _ends_axis(norm_col: str, axis: str) -> bool:
    return re.search(rf"[_\-]{axis}$", norm_col) is not None


def _strip_axis(norm_col: str, axis: str) -> str:
    return re.sub(rf"[_\-]{axis}$", "", norm_col)


def find_xyz_triplet_by_suffix(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Robust fallback: find something_x, something_y, something_z.
    Prefer bases containing head-like keywords, otherwise prefer later columns.
    """
    cols = list(df.columns)
    norms = [normalize_col_for_suffix(c) for c in cols]
    n = len(cols)

    # keywords (work if Japanese decoded; if mojibake, later-column preference helps)
    keywords = ["頭", "head", "face", "center", "中心", "位置", "pos"]

    # 1) adjacent triplet
    best = None  # (score, i, x, y, z)
    for i in range(n - 2):
        nx, ny, nz = norms[i], norms[i + 1], norms[i + 2]
        if _ends_axis(nx, "x") and _ends_axis(ny, "y") and _ends_axis(nz, "z"):
            bx = _strip_axis(nx, "x")
            by = _strip_axis(ny, "y")
            bz = _strip_axis(nz, "z")
            if bx and (bx == by == bz):
                score = 0.0
                if any(k in bx for k in keywords):
                    score += 10.0
                score += (i / max(1, n - 1)) * 2.0  # prefer later
                cand = (score, i, cols[i], cols[i + 1], cols[i + 2])
                best = cand if best is None else max(best, cand)
    if best:
        return best[2], best[3], best[4]

    # 2) grouped by base
    groups: Dict[str, Dict[str, str]] = {}
    for c, nc in zip(cols, norms):
        for axis in ("x", "y", "z"):
            if _ends_axis(nc, axis):
                base = _strip_axis(nc, axis)
                if not base:
                    continue
                groups.setdefault(base, {})[axis] = c

    best2 = None  # (score, base, x, y, z)
    for base, d in groups.items():
        if all(a in d for a in ("x", "y", "z")):
            score = 0.0
            if any(k in base for k in keywords):
                score += 10.0
            idxs = [cols.index(d["x"]), cols.index(d["y"]), cols.index(d["z"])]
            score += (max(idxs) / max(1, n - 1)) * 2.0
            cand = (score, base, d["x"], d["y"], d["z"])
            best2 = cand if best2 is None else max(best2, cand)

    if best2:
        return best2[2], best2[3], best2[4]

    return None, None, None


# ============================================================
# SKYCOM azimuth conversion (hard-coded mode)
# ============================================================
def az_to_yaw_zref(azimuth_deg: float) -> float:
    """
    Convert SKYCOM azimuth to yaw where:
      yaw=0 points +Z, yaw=+90 points +X (right)

    Mode used: x0_ccw_to_z  (same as your vis command)
      yaw = wrap180(90 - az)
    """
    a = float(azimuth_deg)
    if HEAD_AZIMUTH_MODE == "x0_ccw_to_z":
        return wrap180(90.0 - a)
    elif HEAD_AZIMUTH_MODE == "x0_cw_to_z":
        return wrap180(90.0 + a)
    elif HEAD_AZIMUTH_MODE == "z0_ccw_to_x":
        return wrap180(a)
    elif HEAD_AZIMUTH_MODE == "z0_cw_to_x":
        return wrap180(-a)
    else:
        return wrap180(a)


def compute_beta_c2p_from_head_xyz(head_x: float, head_y: float, head_z: float) -> float:
    """
    beta_c2p = atan2(right_rel, forward_rel) in degrees
    where right_rel/forward_rel are chosen by LOS_AXES.
    Using head position relative to camera origin.
    """
    # camera -> head vector components
    dx = float(head_x) - float(CAM_X_MM)
    dy = float(head_y) - float(CAM_Y_MM)
    dz = float(head_z) - float(CAM_Z_MM)

    ax = LOS_AXES.lower().strip()
    if len(ax) != 2 or ax[0] == ax[1] or any(a not in "xyz" for a in ax):
        ax = "xz"

    def pick(a: str) -> float:
        if a == "x":
            return dx
        if a == "y":
            return dy
        if a == "z":
            return dz
        return 0.0

    right_rel = pick(ax[0])
    forward_rel = pick(ax[1])
    beta = np.degrees(np.arctan2(right_rel, forward_rel))
    return float(wrap180(beta))


def compute_yaw_p2c_from_beta(beta_c2p: float) -> float:
    """
    yaw_p2c = wrap180( LOS_YAW_SIGN * (beta_c2p + 180) + LOS_YAW_OFFSET_DEG )
    """
    return float(wrap180(float(LOS_YAW_SIGN) * (float(beta_c2p) + 180.0) + float(LOS_YAW_OFFSET_DEG)))


# ============================================================
# Args (correction knobs are NOT here; hard-coded)
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--aisin_root", required=True)
    p.add_argument("--output_dir", required=True)

    # these are column names in CSV (keep as args)
    p.add_argument("--yaw_col", default="頭方位角")
    p.add_argument("--pitch_col", default="頭仰角")
    p.add_argument("--image_id_col", default="ImageID")
    p.add_argument("--csv_skip_rows", type=int, default=11)

    p.add_argument("--output_format", choices=["aflw", "npz", "both"], default="aflw")

    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--crop_margin", type=float, default=0.4)
    p.add_argument("--on_no_face", choices=["skip", "center", "error"], default="skip")
    p.add_argument("--det_size", type=int, default=640)

    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


# ============================================================
# Face detection (CPU only to avoid CUDA lib issues)
# ============================================================
def init_face_detector(det_size: int):
    if not INSIGHTFACE_AVAILABLE:
        return None
    # CPU only (avoid libcublasLt.so missing)
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def detect_and_crop_face(img_bgr, detector, margin, size):
    faces = detector.get(img_bgr)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = face.bbox.astype(int)
    w, h = x2-x1, y2-y1
    m = int(max(w, h) * margin)
    x1, y1 = max(0, x1-m), max(0, y1-m)
    x2, y2 = min(img_bgr.shape[1], x2+m), min(img_bgr.shape[0], y2+m)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def center_crop_resize(img_bgr, size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = img_bgr[y0:y0+s, x0:x0+s]
    return cv2.resize(crop, (size, size))


# ============================================================
# CSV loader (now loads head xyz too)
# ============================================================
def detect_head_xyz_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    global HEAD_X_COL, HEAD_Y_COL, HEAD_Z_COL

    # 1) hardcoded override (if you set HEAD_X_COL etc at top)
    if HEAD_X_COL and HEAD_Y_COL and HEAD_Z_COL:
        if HEAD_X_COL in df.columns and HEAD_Y_COL in df.columns and HEAD_Z_COL in df.columns:
            return HEAD_X_COL, HEAD_Y_COL, HEAD_Z_COL

    # 2) try keyword-based detection
    x_col = find_column(df, ["頭_x", "頭x", "頭部_x", "頭部x", "頭位置_x", "頭位置x", "head_x", "headx", "headpos_x", "headposx", "頭中心_x", "頭中心x"])
    y_col = find_column(df, ["頭_y", "頭y", "頭部_y", "頭部y", "頭位置_y", "頭位置y", "head_y", "heady", "headpos_y", "headposy", "頭中心_y", "頭中心y"])
    z_col = find_column(df, ["頭_z", "頭z", "頭部_z", "頭部z", "頭位置_z", "頭位置z", "head_z", "headz", "headpos_z", "headposz", "頭中心_z", "頭中心z"])

    if x_col and y_col and z_col:
        return x_col, y_col, z_col

    # 3) robust suffix triplet detection
    tx, ty, tz = find_xyz_triplet_by_suffix(df)
    if tx and ty and tz:
        return tx, ty, tz

    raise RuntimeError("Failed to auto-detect head x/y/z columns. Please set HEAD_X_COL/HEAD_Y_COL/HEAD_Z_COL at top or adjust candidates.")


def load_motion_csv(path: Path, skip: int, yaw_col: str, pitch_col: str, image_id_col: str) -> pd.DataFrame:
    # IMPORTANT: try UTF-8 first to avoid mojibake when possible
    df = None
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift-jis", "euc-jp"]:
        try:
            df = pd.read_csv(path, skiprows=skip, encoding=enc)
            last_err = None
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Failed to read CSV: {path} (last_err={last_err})")

    df = df.rename(columns=str.strip)

    # detect head xyz columns
    hx, hy, hz = detect_head_xyz_columns(df)

    # keep needed cols
    keep = [image_id_col, yaw_col, pitch_col, hx, hy, hz]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {path.name}: {missing}")

    df = df[keep].dropna()

    df[image_id_col] = df[image_id_col].astype(int)
    # rename to stable internal names
    return df.rename(columns={
        image_id_col: "image_id",
        yaw_col: "azimuth",
        pitch_col: "pitch",
        hx: "head_x",
        hy: "head_y",
        hz: "head_z",
    })


# ============================================================
# Main processing
# ============================================================
def main():
    args = parse_args()
    setup_logging(args.verbose)

    detector = init_face_detector(args.det_size)
    if detector is None:
        logging.error("InsightFace not available")
        sys.exit(1)

    stats = ConversionStats()
    samples: List[Dict[str, object]] = []

    motion_dir = Path(args.aisin_root) / "motion"
    image_dir = Path(args.aisin_root) / "image"

    for csv_file in sorted(motion_dir.glob("*_解析結果ファイル.csv")):
        movie_id = csv_file.stem.split("_")[0]
        stats.total_movies += 1

        try:
            df = load_motion_csv(
                csv_file, args.csv_skip_rows,
                args.yaw_col, args.pitch_col, args.image_id_col
            )
        except Exception as e:
            stats.processing_errors += 1
            logging.warning(f"[SKIP] CSV load failed: {csv_file.name}: {e}")
            continue

        stats.total_csv_rows += len(df)

        # build image map
        img_map: Dict[int, Path] = {}
        for folder in image_dir.glob(f"{int(movie_id):02d}_*"):
            for p in folder.glob("*.png"):
                try:
                    iid = int(p.stem.split("_")[-1])
                    img_map[iid] = p
                except Exception:
                    pass

        stats.total_images_found += len(img_map)

        for _, r in df.iterrows():
            try:
                iid = int(r["image_id"])
                if iid not in img_map:
                    stats.unmatched_csv_rows += 1
                    continue

                img = cv2.imread(str(img_map[iid]))
                if img is None:
                    stats.processing_errors += 1
                    continue

                face = detect_and_crop_face(img, detector, args.crop_margin, args.img_size)
                if face is None:
                    stats.face_detection_failed += 1
                    if args.on_no_face == "skip":
                        continue
                    elif args.on_no_face == "center":
                        face = center_crop_resize(img, args.img_size)
                    else:
                        raise RuntimeError("No face detected")
                else:
                    stats.face_detection_success += 1

                # ---------------------------
                # HARD-CODED yaw correction
                # ---------------------------
                az = float(r["azimuth"])
                pitch = float(r["pitch"])
                head_x = float(r["head_x"])
                head_y = float(r["head_y"])
                head_z = float(r["head_z"])

                # base yaw from azimuth
                yaw_conv = az_to_yaw_zref(az)
                yaw_base = wrap180(float(YAW_SIGN) * float(yaw_conv) + float(YAW_OFFSET_DEG))

                # pitch (simple)
                pitch_out = wrap180(float(PITCH_SIGN) * float(pitch) + float(PITCH_OFFSET_DEG))

                # LOS correction using head xyz on xz plane
                beta_c2p = compute_beta_c2p_from_head_xyz(head_x, head_y, head_z)
                yaw_p2c = compute_yaw_p2c_from_beta(beta_c2p)

                yaw_out = wrap180(yaw_base - yaw_p2c)

                samples.append({
                    "image": cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
                    "yaw": float(yaw_out),
                    "pitch": float(pitch_out),
                    "roll": 0.0,
                    "movie_id": str(movie_id),
                    "image_id": int(iid),
                })
                stats.matched_pairs += 1

            except Exception as e:
                stats.processing_errors += 1
                if args.verbose:
                    logging.exception(f"Row processing error (movie={movie_id}): {e}")
                continue

    if len(samples) == 0:
        logging.error("No samples generated.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Save AFLW =====
    if args.output_format in ["aflw", "both"]:
        img_dir = out_dir / "images"
        img_dir.mkdir(exist_ok=True)
        files = []

        for i, s in enumerate(samples):
            name = f"image_{i:05d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"),
                        cv2.cvtColor(s["image"], cv2.COLOR_RGB2BGR))  # type: ignore[arg-type]
            with open(img_dir / f"{name}.txt", "w") as f:
                f.write(
                    f"0 {np.deg2rad(float(s['yaw'])):.6f} "
                    f"{np.deg2rad(float(s['pitch'])):.6f} 0.000000\n"
                )
            files.append(f"images/{name}")

        with open(out_dir / "files.txt", "w") as f:
            f.write("\n".join(files))

    # ===== Save NPZ =====
    if args.output_format in ["npz", "both"]:
        images = np.array([s["image"] for s in samples], dtype=np.uint8)  # type: ignore[list-item]
        poses = np.array([[float(s["yaw"]), float(s["pitch"]), 0.0] for s in samples], dtype=np.float32)
        np.savez(out_dir / "aisin.npz", image=images, pose=poses)

    # ===== Meta =====
    meta = {
        "arguments": vars(args),
        "hardcoded_correction": {
            "CAM_X_MM": CAM_X_MM, "CAM_Y_MM": CAM_Y_MM, "CAM_Z_MM": CAM_Z_MM,
            "LOS_AXES": LOS_AXES,
            "HEAD_AZIMUTH_MODE": HEAD_AZIMUTH_MODE,
            "YAW_SIGN": YAW_SIGN, "YAW_OFFSET_DEG": YAW_OFFSET_DEG,
            "LOS_YAW_SIGN": LOS_YAW_SIGN, "LOS_YAW_OFFSET_DEG": LOS_YAW_OFFSET_DEG,
            "PITCH_SIGN": PITCH_SIGN, "PITCH_OFFSET_DEG": PITCH_OFFSET_DEG,
            "HEAD_X_COL": HEAD_X_COL, "HEAD_Y_COL": HEAD_Y_COL, "HEAD_Z_COL": HEAD_Z_COL,
        },
        "statistics": asdict(stats),
        "timestamp": datetime.now().isoformat()
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(stats.report())
    logging.info("Conversion complete.")


if __name__ == "__main__":
    main()

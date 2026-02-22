#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root()))
from acl_predictor import AclPredictor  # noqa: E402
from scripts.common_vis_utils import (  # noqa: E402
    _annotate_side_by_side,
    _make_side_by_side,
    draw_rotated_boxes,
    preprocess_image,
)


def _parse_hw(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"invalid input-hw: {s}")
    return int(parts[0]), int(parts[1])


def _list_images(d: Path) -> List[Path]:
    exts = {".bmp", ".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _preprocess(img_bgr: np.ndarray, input_hw: Tuple[int, int]) -> np.ndarray:
    expected_h, expected_w = input_hw
    resized = cv2.resize(img_bgr, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, 0).copy(order="C")


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        return float("inf")
    return float(np.max(np.abs(aa - bb)))


def _discover_ort_cache_pairs(cache_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for inp in cache_root.rglob("inputs_from_ort.npz"):
        out = inp.parent / "ort_final_outputs.npz"
        if out.is_file():
            pairs.append((inp, out))
    return pairs


def _build_cached_ort_map(
    image_tensors: Dict[str, np.ndarray],
    pairs: List[Tuple[Path, Path]],
    match_eps: float = 1e-6,
) -> Dict[str, Dict[str, np.ndarray]]:
    cached: Dict[str, Dict[str, np.ndarray]] = {}
    if not pairs:
        return cached
    img_items = list(image_tensors.items())
    for inp_npz, out_npz in pairs:
        try:
            inp = np.load(inp_npz)["input"].astype(np.float32)
            outs = np.load(out_npz)
            dets = np.asarray(outs["dets"])
            labels = np.asarray(outs["labels"])
        except Exception:
            continue

        best_name: Optional[str] = None
        best_diff = float("inf")
        for name, x in img_items:
            if x.shape != inp.shape:
                continue
            d = float(np.max(np.abs(x - inp)))
            if d < best_diff:
                best_diff = d
                best_name = name
        if best_name is not None and best_diff <= match_eps and best_name not in cached:
            cached[best_name] = {
                "dets": dets,
                "labels": labels,
                "input_max_abs": np.float32(best_diff),
                "src_inp_npz": str(inp_npz),
                "src_out_npz": str(out_npz),
            }
    return cached


def _percentile_ms(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def main() -> int:
    root = _repo_root()
    p = argparse.ArgumentParser()
    p.add_argument("--om", required=True)
    p.add_argument("--image-dir", default=str(root / "images"))
    p.add_argument("--input-hw", default="672,1024")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--cache-root", default=str(root / "output"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--single-image", default="", help="For benchmark; default first selected image")
    p.add_argument("--bench-warmup", type=int, default=10)
    p.add_argument("--bench-iters", type=int, default=3)
    args = p.parse_args()

    om_path = Path(args.om).resolve()
    if not om_path.is_file():
        raise SystemExit(f"OM not found: {om_path}")

    image_dir = Path(args.image_dir).resolve()
    images = _list_images(image_dir)
    if not images:
        raise SystemExit(f"no images under: {image_dir}")
    if args.limit > 0:
        images = images[: int(args.limit)]
    hw = _parse_hw(args.input_hw)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: Dict[str, np.ndarray] = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        tensors[img_path.name] = _preprocess(img, hw).astype(np.float32)
    if not tensors:
        raise SystemExit("all images failed to read")

    cache_pairs = _discover_ort_cache_pairs(Path(args.cache_root).resolve())
    cached_map = _build_cached_ort_map(tensors, cache_pairs, match_eps=1e-6)

    predictor = AclPredictor(model_path=str(om_path), device_id=int(args.device_id), do_finalize=True)

    rows = []
    om_lat_ms: List[float] = []
    compared = 0
    for img_path in images:
        x = tensors.get(img_path.name, None)
        if x is None:
            continue
        t0 = time.perf_counter()
        dets_om, labels_om = predictor.run(x)[:2]
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000.0
        om_lat_ms.append(lat)

        dets_om = np.asarray(dets_om)
        labels_om = np.asarray(labels_om)

        per_dir = out_dir / img_path.stem
        per_dir.mkdir(parents=True, exist_ok=True)
        np.save(per_dir / "input.npy", x)
        np.save(per_dir / "dets_om.npy", dets_om)
        np.save(per_dir / "labels_om.npy", labels_om)

        dets_max_abs = float("nan")
        dets_mean_abs = float("nan")
        dets_num_diff_gt_1e_3 = -1
        labels_neq = -1
        ort_src = ""
        if img_path.name in cached_map:
            cached = cached_map[img_path.name]
            dets_ort = np.asarray(cached["dets"])
            labels_ort = np.asarray(cached["labels"])
            d = np.abs(np.asarray(dets_ort, dtype=np.float32) - np.asarray(dets_om, dtype=np.float32))
            dets_max_abs = float(np.max(d))
            dets_mean_abs = float(np.mean(d))
            dets_num_diff_gt_1e_3 = int(np.sum(d > 1e-3))
            labels_neq = int(np.sum(np.asarray(labels_ort).astype(np.int64) != np.asarray(labels_om).astype(np.int64)))
            np.save(per_dir / "dets_ort.npy", dets_ort)
            np.save(per_dir / "labels_ort.npy", labels_ort)
            ort_src = cached["src_out_npz"]
            compared += 1

        # Always emit visualization files for each image directory.
        img = cv2.imread(str(img_path))
        if img is not None:
            _, vis_base = preprocess_image(img, hw)
            vis_om = draw_rotated_boxes(vis_base, dets_om, labels_om, 0.3)
            if img_path.name in cached_map:
                cached = cached_map[img_path.name]
                dets_ort = np.asarray(cached["dets"])
                labels_ort = np.asarray(cached["labels"])
                vis_ort = draw_rotated_boxes(vis_base, dets_ort, labels_ort, 0.3)
                side = _make_side_by_side(vis_ort, vis_om)
                side = _annotate_side_by_side(side, "ORT(cache)", "NPU(OM)")
            else:
                vis_ort = vis_base.copy()
                cv2.putText(
                    vis_ort,
                    "ORT cache missing",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                side = _make_side_by_side(vis_ort, vis_om)
                side = _annotate_side_by_side(side, "ORT(missing)", "NPU(OM)")
            cv2.imwrite(str(per_dir / "vis_input.jpg"), vis_base)
            cv2.imwrite(str(per_dir / "vis_ort.jpg"), vis_ort)
            cv2.imwrite(str(per_dir / "vis_om.jpg"), vis_om)
            cv2.imwrite(str(per_dir / "vis_side_by_side.jpg"), side)

        rows.append(
            {
                "image": img_path.name,
                "om_ms": f"{lat:.3f}",
                "dets_max_abs": f"{dets_max_abs:.6g}" if np.isfinite(dets_max_abs) else "N/A",
                "dets_mean_abs": f"{dets_mean_abs:.6g}" if np.isfinite(dets_mean_abs) else "N/A",
                "dets_num_diff_gt_1e-3": str(dets_num_diff_gt_1e_3 if dets_num_diff_gt_1e_3 >= 0 else "N/A"),
                "labels_neq": str(labels_neq if labels_neq >= 0 else "N/A"),
                "ort_cache_src": ort_src,
            }
        )

    csv_path = out_dir / "summary_om_only.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "om_ms",
                "dets_max_abs",
                "dets_mean_abs",
                "dets_num_diff_gt_1e-3",
                "labels_neq",
                "ort_cache_src",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # single-image benchmark
    bench_image_name = str(args.single_image).strip()
    if not bench_image_name:
        bench_image_name = rows[0]["image"] if rows else ""
    if not bench_image_name:
        raise SystemExit("no image for benchmark")
    if bench_image_name not in tensors:
        raise SystemExit(f"benchmark image not in selected set: {bench_image_name}")
    bench_x = tensors[bench_image_name]
    warmup = max(0, int(args.bench_warmup))
    iters = max(1, int(args.bench_iters))
    for _ in range(warmup):
        predictor.run(bench_x)
    bench_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        predictor.run(bench_x)
        t1 = time.perf_counter()
        bench_ms.append((t1 - t0) * 1000.0)

    report = {
        "om": str(om_path),
        "image_dir": str(image_dir),
        "selected_images": [str(p.name) for p in images if p.name in tensors],
        "selected_count": int(len(tensors)),
        "ort_cache_pair_count": int(len(cache_pairs)),
        "matched_ort_cache_count": int(len(cached_map)),
        "compared_count": int(compared),
        "latency_ms": {
            "mean": float(np.mean(np.asarray(om_lat_ms, dtype=np.float64))) if om_lat_ms else 0.0,
            "p50": _percentile_ms(om_lat_ms, 50),
            "p90": _percentile_ms(om_lat_ms, 90),
            "max": max(om_lat_ms) if om_lat_ms else 0.0,
            "min": min(om_lat_ms) if om_lat_ms else 0.0,
        },
        "single_image_bench": {
            "image": bench_image_name,
            "warmup": warmup,
            "iters": iters,
            "mean_ms": float(np.mean(np.asarray(bench_ms, dtype=np.float64))),
            "p50_ms": _percentile_ms(bench_ms, 50),
            "p90_ms": _percentile_ms(bench_ms, 90),
            "p99_ms": _percentile_ms(bench_ms, 99),
            "max_ms": max(bench_ms),
            "min_ms": min(bench_ms),
            "fps_from_mean": float(1000.0 / np.mean(np.asarray(bench_ms, dtype=np.float64))),
        },
        "artifacts": {
            "summary_csv": str(csv_path),
        },
    }
    report_path = out_dir / "report_om_only.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

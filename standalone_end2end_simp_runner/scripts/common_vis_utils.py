#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def preprocess_image(img_bgr: np.ndarray, input_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    expected_h, expected_w = input_hw
    resized = cv2.resize(img_bgr, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))
    nchw = np.expand_dims(chw, 0).copy(order="C")
    return nchw, resized


def _rotated_box_points(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    wx, wy, hx, hy = 0.5 * np.array([w, w, -h, h], dtype=np.float32) * np.array(
        [np.cos(angle), np.sin(angle), np.sin(angle), np.cos(angle)], dtype=np.float32
    )
    pts = np.array(
        [
            [cx - wx - hx, cy - wy - hy],
            [cx + wx - hx, cy + wy - hy],
            [cx + wx + hx, cy + wy + hy],
            [cx - wx + hx, cy - wy + hy],
        ],
        dtype=np.float32,
    )
    return pts


def draw_rotated_boxes(img_bgr: np.ndarray, dets: np.ndarray, labels: np.ndarray, score_thr: float) -> np.ndarray:
    img = img_bgr.copy()
    dets = np.asarray(dets)
    labels = np.asarray(labels)
    if dets.ndim == 3:
        dets = dets[0]
    if labels.ndim == 2:
        labels = labels[0]
    for det, label in zip(dets, labels):
        if det.shape[0] < 6:
            continue
        score = float(det[5])
        if score < score_thr:
            continue
        cx, cy, w, h, angle = [float(x) for x in det[0:5]]
        pts = _rotated_box_points(cx, cy, w, h, angle)
        pts_i = np.round(pts).astype(np.int32).reshape(1, 4, 2)
        cv2.drawContours(img, pts_i, -1, (0, 255, 0), 2)
        label_int = int(label)
        text = f"cls{label_int} {score:.2f}"
        org = (max(int(pts_i[0, 0, 0]), 0), max(int(pts_i[0, 0, 1]) - 3, 0))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def _make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    h = max(lh, rh)

    def _pad(img: np.ndarray, h: int, w: int) -> np.ndarray:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[: img.shape[0], : img.shape[1]] = img
        return out

    left2 = _pad(left, h, lw)
    right2 = _pad(right, h, rw)
    return np.concatenate([left2, right2], axis=1)


def _annotate_side_by_side(img: np.ndarray, left_label: str, right_label: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    mid = w // 2

    top_h = max(28, int(h * 0.06))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (mid - 1, top_h), (255, 255, 255), thickness=-1)
    cv2.rectangle(overlay, (mid, 0), (w - 1, top_h), (255, 255, 255), thickness=-1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    pad_x = 10
    base_y = int(top_h * 0.72)

    def put(x: int, text: str) -> None:
        cv2.putText(out, text, (x, base_y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, text, (x, base_y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

    put(pad_x, left_label)
    put(mid + pad_x, right_label)

    cv2.rectangle(out, (0, 0), (mid - 1, h - 1), (0, 0, 0), thickness=2)
    cv2.rectangle(out, (mid, 0), (w - 1, h - 1), (0, 0, 0), thickness=2)
    cv2.line(out, (mid, 0), (mid, h - 1), (0, 0, 0), thickness=2)
    return out

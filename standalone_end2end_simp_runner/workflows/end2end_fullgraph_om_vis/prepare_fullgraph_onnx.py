#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../standalone_end2end_simp_runner/workflows/end2end_fullgraph_om_vis
    return Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(
        description="Prepare fullgraph ONNX for ATC: atc_ready_full + optional Where/Gather rewrites."
    )
    ap.add_argument("--in-onnx", default=str(root / "ascend_c_onnx" / "end2end_simp.onnx"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--enable-where2-maskarith", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--where-name", default="/Where_2")
    ap.add_argument("--pi2-name", default="", help="Optional constant initializer for where->maskarith rewrite.")
    ap.add_argument("--enable-gather40-slice", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gather-name", default="/Gather_40")
    ap.add_argument(
        "--decompose-npu-layernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decompose NPULayerNorm nodes back to standard ops to minimize required custom ops.",
    )
    args = ap.parse_args()

    in_onnx = Path(args.in_onnx).resolve()
    if not in_onnx.is_file():
        raise SystemExit(f"missing --in-onnx: {in_onnx}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = out_dir / "npu_full.atc_ready_full.onnx"
    cmd = [
        sys.executable,
        str(root / "scripts" / "patch_end2end_simp_pipeline.py"),
        "--pipeline",
        "atc_ready_full",
        "--in-onnx",
        str(in_onnx),
        "--out-onnx",
        str(base),
    ]
    if bool(args.decompose_npu_layernorm):
        cmd.append("--decompose-npu-layernorm")
    _run(cmd)

    current = base
    steps: list[dict[str, str]] = [
        {"step": "atc_ready_full", "in": str(in_onnx), "out": str(base)},
    ]

    if bool(args.enable_where2_maskarith):
        out1 = out_dir / "npu_full.atc_ready_full.where2_maskarith.onnx"
        cmd = [
            sys.executable,
            str(root / "tools" / "patch_where2_maskarith.py"),
            "--in-onnx",
            str(current),
            "--out-onnx",
            str(out1),
            "--where-name",
            str(args.where_name),
        ]
        if str(args.pi2_name).strip():
            cmd += ["--pi2-name", str(args.pi2_name).strip()]
        _run(cmd)
        steps.append({"step": "where2_maskarith", "in": str(current), "out": str(out1)})
        current = out1

    if bool(args.enable_gather40_slice):
        suffix = "where2_maskarith.gather40_slice" if "where2_maskarith" in current.name else "gather40_slice"
        out2 = out_dir / f"npu_full.atc_ready_full.{suffix}.onnx"
        _run(
            [
                sys.executable,
                str(root / "tools" / "patch_gather40_to_slice_squeeze.py"),
                "--in-onnx",
                str(current),
                "--out-onnx",
                str(out2),
                "--gather-name",
                str(args.gather_name),
            ]
        )
        steps.append({"step": "gather40_slice_squeeze", "in": str(current), "out": str(out2)})
        current = out2

    manifest = {
        "repo_root": str(root),
        "input_onnx": str(in_onnx),
        "final_onnx": str(current),
        "steps": steps,
    }
    manifest_path = out_dir / "fullgraph_prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"FINAL_ONNX={current}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

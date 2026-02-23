#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper, shape_inference
import numpy as np


def _find_node_index(model: onnx.ModelProto, name: str) -> int:
    for i, n in enumerate(model.graph.node):
        if n.name == name:
            return i
    raise RuntimeError(f"node not found: {name}")


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for a in node.attribute:
        if a.name == name:
            return int(onnx.helper.get_attribute_value(a))
    return int(default)


def _build_shape_map(model: onnx.ModelProto) -> dict[str, list[int | str | None]]:
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    out: dict[str, list[int | str | None]] = {}
    vis = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for vi in vis:
        if not vi.type.HasField("tensor_type"):
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims: list[int | str | None] = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            elif d.HasField("dim_param"):
                dims.append(str(d.dim_param))
            else:
                dims.append(None)
        out[vi.name] = dims
    return out


def _find_target_gather(
    model: onnx.ModelProto,
    shape_map: dict[str, list[int | str | None]],
    axis: int,
    scalar_index: int,
    last_dim: int,
) -> int:
    init_map = {x.name: x for x in model.graph.initializer}
    cands: list[tuple[int, onnx.NodeProto]] = []
    for idx, n in enumerate(model.graph.node):
        if n.op_type != "Gather" or len(n.input) != 2 or len(n.output) != 1:
            continue
        data_name, index_name = n.input
        g_axis = _attr_int(n, "axis", 0)
        if g_axis != int(axis):
            continue
        if index_name not in init_map:
            continue
        idx_arr = numpy_helper.to_array(init_map[index_name])
        if not (idx_arr.size == 1 and int(idx_arr.reshape(-1)[0]) == int(scalar_index)):
            continue
        data_shape = shape_map.get(data_name)
        if not data_shape or not isinstance(data_shape[-1], int):
            continue
        if int(data_shape[-1]) != int(last_dim):
            continue
        cands.append((idx, n))
    if not cands:
        raise RuntimeError(
            f"no Gather matches condition: axis={axis}, scalar_index={scalar_index}, last_dim={last_dim}"
        )
    if len(cands) > 1:
        names = [x[1].name for x in cands]
        raise RuntimeError(f"multiple Gather nodes match condition, please pass --gather-name explicitly: {names}")
    return int(cands[0][0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Replace Gather(axis=1, scalar index=-1, input last dim=5) "
            "with Slice(axis=1,4:5)+Squeeze."
        )
    )
    ap.add_argument("--in-onnx", required=True)
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument(
        "--gather-name",
        default="",
        help="Optional explicit Gather node name. If empty, auto-find by axis/index/last-dim condition.",
    )
    ap.add_argument("--target-axis", type=int, default=1)
    ap.add_argument("--target-index", type=int, default=-1)
    ap.add_argument("--target-last-dim", type=int, default=5)
    args = ap.parse_args()

    in_onnx = Path(args.in_onnx).resolve()
    out_onnx = Path(args.out_onnx).resolve()
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(in_onnx))
    shape_map = _build_shape_map(model)

    gather_name = str(args.gather_name).strip()
    if gather_name:
        idx = _find_node_index(model, gather_name)
    else:
        idx = _find_target_gather(
            model,
            shape_map,
            axis=int(args.target_axis),
            scalar_index=int(args.target_index),
            last_dim=int(args.target_last_dim),
        )

    g = model.graph.node[idx]
    if g.op_type != "Gather":
        raise RuntimeError(f"{g.name} op_type={g.op_type}, expected Gather")
    if len(g.input) != 2 or len(g.output) != 1:
        raise RuntimeError(f"unexpected Gather io for {g.name}")

    data_name, index_name = g.input
    out_name = g.output[0]
    axis = _attr_int(g, "axis", 0)
    if axis != int(args.target_axis):
        raise RuntimeError(f"{g.name} axis={axis}, expected {args.target_axis}")

    init_map = {x.name: x for x in model.graph.initializer}
    if index_name not in init_map:
        raise RuntimeError(f"gather index initializer missing: {index_name}")
    idx_arr = numpy_helper.to_array(init_map[index_name])
    if not (idx_arr.size == 1 and int(idx_arr.reshape(-1)[0]) == int(args.target_index)):
        raise RuntimeError(f"expected scalar -1 index, got: {idx_arr}")

    data_shape = shape_map.get(data_name)
    if not data_shape or not isinstance(data_shape[-1], int):
        raise RuntimeError(f"{g.name} input shape missing or dynamic on last dim: {data_shape}")
    if int(data_shape[-1]) != int(args.target_last_dim):
        raise RuntimeError(f"{g.name} input last dim={data_shape[-1]}, expected {args.target_last_dim}")

    starts_v = int(args.target_last_dim) - 1
    ends_v = int(args.target_last_dim)

    base = g.name if g.name else f"Gather_{idx}"
    starts_name = f"{base}_slice_starts"
    ends_name = f"{base}_slice_ends"
    axes_name = f"{base}_slice_axes"
    steps_name = f"{base}_slice_steps"
    slice_out = f"{base}_slice_output_0"

    new_inits = [
        numpy_helper.from_array(np.asarray([starts_v], dtype=np.int64), name=starts_name),
        numpy_helper.from_array(np.asarray([ends_v], dtype=np.int64), name=ends_name),
        numpy_helper.from_array(np.asarray([axis], dtype=np.int64), name=axes_name),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=steps_name),
    ]
    model.graph.initializer.extend(new_inits)

    slice_node = helper.make_node(
        "Slice",
        inputs=[data_name, starts_name, ends_name, axes_name, steps_name],
        outputs=[slice_out],
        name=f"{base}_SliceLastCol",
    )
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=[slice_out],
        outputs=[out_name],
        name=f"{base}_SqueezeLastCol",
        axes=[axis],
    )

    del model.graph.node[idx]
    model.graph.node.insert(idx, slice_node)
    model.graph.node.insert(idx + 1, squeeze_node)

    try:
        onnx.checker.check_model(model, check_custom_domain=False)
    except Exception as e:
        print(f"[WARN] skip strict checker due to custom op schema: {e}")

    onnx.save(model, str(out_onnx))
    print(f"[OK] in={in_onnx}")
    print(f"[OK] out={out_onnx}")
    print(
        f"[OK] replaced {g.name}: Gather(axis={axis}, index={args.target_index}, "
        f"last_dim={args.target_last_dim}) -> Slice({starts_v}:{ends_v})+Squeeze"
    )


if __name__ == "__main__":
    main()

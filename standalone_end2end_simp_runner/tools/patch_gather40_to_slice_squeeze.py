#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replace /Gather_40(axis=1,index=-1) with Slice(axis=1,4:5)+Squeeze."
    )
    ap.add_argument("--in-onnx", required=True)
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument("--gather-name", default="/Gather_40")
    args = ap.parse_args()

    in_onnx = Path(args.in_onnx).resolve()
    out_onnx = Path(args.out_onnx).resolve()
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(in_onnx))
    idx = _find_node_index(model, args.gather_name)
    g = model.graph.node[idx]
    if g.op_type != "Gather":
        raise RuntimeError(f"{args.gather_name} op_type={g.op_type}, expected Gather")
    if len(g.input) != 2 or len(g.output) != 1:
        raise RuntimeError(f"unexpected Gather io for {args.gather_name}")

    data_name, index_name = g.input
    out_name = g.output[0]
    axis = _attr_int(g, "axis", 0)
    if axis != 1:
        raise RuntimeError(f"{args.gather_name} axis={axis}, expected 1")

    init_map = {x.name: x for x in model.graph.initializer}
    if index_name not in init_map:
        raise RuntimeError(f"gather index initializer missing: {index_name}")
    idx_arr = numpy_helper.to_array(init_map[index_name])
    if not (idx_arr.size == 1 and int(idx_arr.reshape(-1)[0]) == -1):
        raise RuntimeError(f"expected scalar -1 index, got: {idx_arr}")

    starts_name = f"{args.gather_name}_slice_starts"
    ends_name = f"{args.gather_name}_slice_ends"
    axes_name = f"{args.gather_name}_slice_axes"
    steps_name = f"{args.gather_name}_slice_steps"
    slice_out = f"{args.gather_name}_slice_output_0"

    new_inits = [
        numpy_helper.from_array(np.asarray([4], dtype=np.int64), name=starts_name),
        numpy_helper.from_array(np.asarray([5], dtype=np.int64), name=ends_name),
        numpy_helper.from_array(np.asarray([axis], dtype=np.int64), name=axes_name),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=steps_name),
    ]
    model.graph.initializer.extend(new_inits)

    slice_node = helper.make_node(
        "Slice",
        inputs=[data_name, starts_name, ends_name, axes_name, steps_name],
        outputs=[slice_out],
        name=f"{args.gather_name}_SliceLastCol",
    )
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=[slice_out],
        outputs=[out_name],
        name=f"{args.gather_name}_SqueezeLastCol",
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
    print(f"[OK] replaced {args.gather_name}: Gather(-1) -> Slice(4:5)+Squeeze")


if __name__ == "__main__":
    main()

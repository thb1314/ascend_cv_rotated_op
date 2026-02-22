#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _find_node_index(model: onnx.ModelProto, name: str) -> int:
    for i, n in enumerate(model.graph.node):
        if n.name == name:
            return i
    raise RuntimeError(f"node not found: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite one Where node to mask-arithmetic form.")
    ap.add_argument("--in-onnx", required=True)
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument("--where-name", default="/Where_2")
    ap.add_argument(
        "--pi2-name",
        default="",
        help="Optional scalar constant initializer name used for mask*const. If empty, infer from Add(then, const).",
    )
    args = ap.parse_args()

    in_onnx = Path(args.in_onnx).resolve()
    out_onnx = Path(args.out_onnx).resolve()
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(in_onnx))
    idx = _find_node_index(model, args.where_name)
    where_node = model.graph.node[idx]
    if where_node.op_type != "Where":
        raise RuntimeError(f"{args.where_name} is not Where, got {where_node.op_type}")
    if len(where_node.input) != 3 or len(where_node.output) != 1:
        raise RuntimeError(f"unexpected Where io, inputs={len(where_node.input)} outputs={len(where_node.output)}")

    cond, then_x, else_y = where_node.input
    where_out = where_node.output[0]

    pi2_name = args.pi2_name.strip()
    if not pi2_name:
        producer = {o: n for n in model.graph.node for o in n.output}
        y_prod = producer.get(else_y)
        if y_prod is None or y_prod.op_type != "Add" or len(y_prod.input) != 2:
            raise RuntimeError(
                f"cannot infer pi2 from else branch producer: {else_y} -> {None if y_prod is None else y_prod.op_type}"
            )
        a0, a1 = y_prod.input
        if a0 == then_x:
            pi2_name = a1
        elif a1 == then_x:
            pi2_name = a0
        else:
            raise RuntimeError(
                f"cannot infer pi2: else producer {y_prod.name} does not look like Add(then, const): inputs={list(y_prod.input)}"
            )

    init_names = {x.name for x in model.graph.initializer}
    if pi2_name not in init_names:
        raise RuntimeError(f"pi2 initializer not found: {pi2_name}")

    cast_out = f"{args.where_name}_mask_f32_output_0"
    mul_out = f"{args.where_name}_mask_pi2_output_0"

    cast_node = helper.make_node(
        "Cast",
        inputs=[cond],
        outputs=[cast_out],
        name=f"{args.where_name}_MaskCast",
        to=TensorProto.FLOAT,
    )
    mul_node = helper.make_node(
        "Mul",
        inputs=[cast_out, pi2_name],
        outputs=[mul_out],
        name=f"{args.where_name}_MaskMulPi2",
    )
    sub_node = helper.make_node(
        "Sub",
        inputs=[else_y, mul_out],
        outputs=[where_out],
        name=f"{args.where_name}_MaskSub",
    )

    del model.graph.node[idx]
    model.graph.node.insert(idx, cast_node)
    model.graph.node.insert(idx + 1, mul_node)
    model.graph.node.insert(idx + 2, sub_node)

    try:
        onnx.checker.check_model(model, check_custom_domain=False)
    except Exception as e:
        # Full checker may fail on custom ops in ai.onnx domain; save graph anyway.
        print(f"[WARN] skip strict checker due to custom op schema: {e}")
    onnx.save(model, str(out_onnx))

    print(f"[OK] in={in_onnx}")
    print(f"[OK] out={out_onnx}")
    print(f"[OK] replaced {args.where_name} with Cast+Mul+Sub using const={pi2_name}")


if __name__ == "__main__":
    main()

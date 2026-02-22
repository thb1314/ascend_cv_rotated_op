from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dedupe_opset_import(m: onnx.ModelProto) -> None:
    by_domain: dict[str, int] = {}
    for o in m.opset_import:
        dom = str(o.domain)
        ver = int(o.version)
        by_domain[dom] = max(by_domain.get(dom, 0), ver)
    del m.opset_import[:]
    for dom, ver in sorted(by_domain.items(), key=lambda kv: kv[0]):
        m.opset_import.extend([onnx.helper.make_opsetid(dom, int(ver))])


def _prune_unused_opset_import(m: onnx.ModelProto) -> None:
    used: set[str] = set()
    for n in m.graph.node:
        used.add(str(getattr(n, "domain", "") or ""))
    kept = [o for o in m.opset_import if str(o.domain) in used]
    del m.opset_import[:]
    m.opset_import.extend(kept)


def _keep_single_default_domain_opset(m: onnx.ModelProto) -> None:
    kept = [o for o in m.opset_import if o.domain == ""]
    if kept:
        del m.opset_import[:]
        m.opset_import.extend(kept[:1])


def _rank_from_value_info(model: onnx.ModelProto, tensor_name: str) -> int | None:
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name != tensor_name:
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return None
        return len(tt.shape.dim)
    return None


def _drop_attr(node: onnx.NodeProto, name: str) -> None:
    kept = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)


def _add_attribute_string_for_atc(node: onnx.NodeProto) -> None:
    parts = []
    for a in node.attribute:
        if a.name == "attribute":
            continue
        if a.type == onnx.AttributeProto.INT:
            parts.append(f'{{"name":"{a.name}","i":{int(a.i)}}}')
        elif a.type == onnx.AttributeProto.FLOAT:
            parts.append(f'{{"name":"{a.name}","f":{float(a.f)}}}')
        elif a.type == onnx.AttributeProto.FLOATS:
            vals = ",".join(str(float(x)) for x in a.floats)
            parts.append(f'{{"name":"{a.name}","floats":[{vals}]}}')
        elif a.type == onnx.AttributeProto.INTS:
            vals = ",".join(str(int(x)) for x in a.ints)
            parts.append(f'{{"name":"{a.name}","ints":[{vals}]}}')
    if not parts:
        return
    _drop_attr(node, "attribute")
    node.attribute.extend([onnx.helper.make_attribute("attribute", "\n".join(parts))])
    if node.op_type == "NPUBatchedRotatedNMS":
        kept = [a for a in node.attribute if a.name == "attribute"]
        del node.attribute[:]
        node.attribute.extend(kept)


def _set_valueinfo_1d_dim(model: onnx.ModelProto, name: str, dim0: int) -> None:
    for vi in list(model.graph.input) + list(model.graph.value_info):
        if vi.name != name:
            continue
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            dims = tt.shape.dim
            del dims[:]
            d0 = tt.shape.dim.add()
            d0.dim_value = int(dim0)


def _reshape_int64_initializer_1d(model: onnx.ModelProto, name: str, dim0: int, fill_value: int) -> None:
    for init in model.graph.initializer:
        if init.name != name:
            continue
        init.data_type = onnx.TensorProto.INT64
        del init.int64_data[:]
        del init.float_data[:]
        del init.double_data[:]
        del init.int32_data[:]
        if hasattr(init, "uint64_data"):
            del init.uint64_data[:]
        if hasattr(init, "uint32_data"):
            del init.uint32_data[:]
        del init.dims[:]
        init.dims.extend([int(dim0)])
        init.raw_data = (int(fill_value).to_bytes(8, byteorder="little", signed=True)) * int(dim0)
        _set_valueinfo_1d_dim(model, name, dim0)
        return


def _patch_grid_priors_hw_inputs(model: onnx.ModelProto) -> None:
    init_map = {i.name: i for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "GridPriorsNPU" or len(node.input) < 3:
            continue
        h_name = node.input[1]
        w_name = node.input[2]
        ih = init_map.get(h_name)
        iw = init_map.get(w_name)
        if ih is None or iw is None:
            continue
        try:
            h_raw = onnx.numpy_helper.to_array(ih).reshape(-1)
            w_raw = onnx.numpy_helper.to_array(iw).reshape(-1)
            h0 = int(h_raw[0]) if h_raw.size else 0
            w0 = int(w_raw[0]) if w_raw.size else 0
            h_dim0 = int(ih.dims[0]) if ih.dims else int(h_raw.size)
            w_dim0 = int(iw.dims[0]) if iw.dims else int(w_raw.size)
            H = h0 if h0 > 0 else h_dim0
            W = w0 if w0 > 0 else w_dim0
        except Exception:
            continue
        if H > 0:
            dim0 = int(ih.dims[0]) if ih.dims else H
            if dim0 > 0:
                _reshape_int64_initializer_1d(model, h_name, dim0, H)
        if W > 0:
            dim0 = int(iw.dims[0]) if iw.dims else W
            if dim0 > 0:
                _reshape_int64_initializer_1d(model, w_name, dim0, W)


def _has_attr(node: onnx.NodeProto, name: str) -> bool:
    return any(a.name == name for a in node.attribute)


def patch_reduce_axes_for_atc(model: onnx.ModelProto) -> int:
    inferred = shape_inference.infer_shapes(model, check_type=False)
    patched = 0
    for n in model.graph.node:
        if n.op_type not in (
            "ReduceMax",
            "ReduceMin",
            "ReduceSum",
            "ReduceMean",
            "ReduceProd",
            "ReduceL2",
            "ReduceLogSumExp",
        ):
            continue
        if _has_attr(n, "axes"):
            continue
        if not n.input:
            continue
        r = _rank_from_value_info(inferred, n.input[0])
        if r is None or r <= 0:
            continue
        axes = list(range(r))
        n.attribute.extend([helper.make_attribute("axes", axes)])
        patched += 1
    return patched


def _patch_topk_to_ascend_topk(model: onnx.ModelProto, opset: int = 11) -> dict:
    inits = {t.name: t for t in model.graph.initializer}
    prod: dict[str, onnx.NodeProto] = {}
    for n in model.graph.node:
        for o in n.output:
            prod[o] = n

    const_cache: dict[str, int] = {}

    def resolve_const_scalar_i64(name: str) -> int:
        if name in const_cache:
            return const_cache[name]
        if name in inits:
            arr = numpy_helper.to_array(inits[name]).reshape(-1)
            if arr.size != 1:
                raise RuntimeError(f"const scalar not scalar: {name} shape={arr.shape}")
            v = int(arr[0])
            const_cache[name] = v
            return v
        n = prod.get(name)
        if n is None:
            raise RuntimeError(f"const scalar producer not found: {name}")
        if n.op_type == "Constant":
            value_attr = None
            for a in n.attribute:
                if a.name == "value":
                    value_attr = a
                    break
            if value_attr is None:
                raise RuntimeError(f"Constant node missing value attribute: {n.name}")
            arr = numpy_helper.to_array(value_attr.t).reshape(-1)
            if arr.size != 1:
                raise RuntimeError(f"Constant value not scalar: {n.name} shape={arr.shape}")
            v = int(arr[0])
            const_cache[name] = v
            return v
        if n.op_type in {"Identity", "Cast", "Squeeze", "Unsqueeze"}:
            if not n.input:
                raise RuntimeError(f"{n.op_type} missing input: {n.name}")
            v = resolve_const_scalar_i64(n.input[0])
            const_cache[name] = v
            return v
        if n.op_type == "Reshape":
            if not n.input:
                raise RuntimeError(f"Reshape missing input: {n.name}")
            v = resolve_const_scalar_i64(n.input[0])
            const_cache[name] = v
            return v
        raise RuntimeError(f"unsupported const scalar op_type={n.op_type} for {name} (node={n.name})")

    def get_k(node: onnx.NodeProto) -> int:
        if len(node.input) < 2:
            raise RuntimeError("TopK node missing K input")
        k_name = node.input[1]
        return int(resolve_const_scalar_i64(k_name))

    def get_attr_i(node: onnx.NodeProto, name: str, default: int) -> int:
        for a in node.attribute:
            if a.name == name:
                return int(a.i)
        return int(default)

    replaced = 0
    skipped_axis = 0
    for n in model.graph.node:
        if n.op_type != "TopK":
            continue
        axis = get_attr_i(n, "axis", 0)
        if axis != 0:
            skipped_axis += 1
            continue
        k = get_k(n)
        largest = get_attr_i(n, "largest", 1)
        sorted_ = get_attr_i(n, "sorted", 1)

        x_in = n.input[0]
        n.ClearField("attribute")
        n.op_type = "AscendTopK"
        n.domain = ""
        del n.input[:]
        n.input.extend([x_in])
        n.attribute.extend(
            [
                onnx.helper.make_attribute("k", int(k)),
                onnx.helper.make_attribute("largest", int(largest)),
                onnx.helper.make_attribute("sorted", int(sorted_)),
            ]
        )
        replaced += 1

    opset_done = False
    for o in model.opset_import:
        if o.domain == "":
            o.version = max(int(o.version), int(opset))
            opset_done = True
    if not opset_done:
        model.opset_import.extend([onnx.helper.make_opsetid("", int(opset))])

    _dedupe_opset_import(model)
    _prune_unused_opset_import(model)
    return {"replaced_topk": replaced, "skipped_axis_neq0": skipped_axis}


def _get_initializer_scalar(model: onnx.ModelProto, name: str) -> np.ndarray:
    for init in model.graph.initializer:
        if init.name == name:
            arr = onnx.numpy_helper.to_array(init)
            return arr
    raise KeyError(f"initializer not found: {name}")


def _infer_rank(model: onnx.ModelProto, tensor_name: str) -> int | None:
    try:
        mi = onnx.shape_inference.infer_shapes(model)
    except Exception:
        mi = model
    for vi in list(mi.graph.input) + list(mi.graph.value_info) + list(mi.graph.output):
        if vi.name != tensor_name:
            continue
        try:
            tt = vi.type.tensor_type
            if not tt.HasField("shape"):
                return None
            return len(tt.shape.dim)
        except Exception:
            return None
    return None


def _patch_nonmaxsuppression_to_npu_nms_ort(
    model: onnx.ModelProto,
    nms_node_name: str = "/NonMaxSuppression",
    boxes_input: str | None = None,
    scores_input: str | None = None,
    max_output_input: str | None = None,
    iou_input: str | None = None,
    score_input: str | None = None,
    swap_boxes_xy: bool = False,
) -> dict:
    nms_idx = None
    for i, n in enumerate(model.graph.node):
        if n.name != nms_node_name:
            continue
        if n.op_type == "NonMaxSuppression":
            nms_idx = i
            break
        if n.op_type == "NPUNonMaxSuppressionOrt":
            _dedupe_opset_import(model)
            _prune_unused_opset_import(model)
            return {"status": "skip", "reason": "already patched"}
    if nms_idx is None:
        _dedupe_opset_import(model)
        _prune_unused_opset_import(model)
        return {"status": "skip", "reason": "node not found"}

    old = model.graph.node[nms_idx]
    if len(old.input) < 2:
        raise SystemExit(f"NonMaxSuppression has too few inputs: {old.name} inputs={list(old.input)}")

    boxes_in = boxes_input or old.input[0]
    scores_in = scores_input or old.input[1]
    max_out_in = max_output_input or (old.input[2] if len(old.input) > 2 else "")
    iou_in = iou_input or (old.input[3] if len(old.input) > 3 else "")
    score_in = score_input or (old.input[4] if len(old.input) > 4 else "")

    def _maybe_get_scalar(name: str, default: float) -> float:
        if not name:
            return float(default)
        try:
            return float(_get_initializer_scalar(model, name).reshape(-1)[0])
        except Exception:
            return float(default)

    max_output = int(_maybe_get_scalar(max_out_in, 3000))
    iou_thr = float(_maybe_get_scalar(iou_in, 0.699999988079071))
    score_thr = float(_maybe_get_scalar(score_in, 0.0))
    out_name = old.output[0]

    new_nodes = []
    swapped_boxes = boxes_in
    if bool(swap_boxes_xy):
        swap_idx_name = "nms_swap_idx_xy"
        if swap_idx_name not in {i.name for i in model.graph.initializer}:
            swap_idx = np.asarray([1, 0, 3, 2], dtype=np.int64)
            model.graph.initializer.append(onnx.numpy_helper.from_array(swap_idx, name=swap_idx_name))
        axis = 2
        r = _infer_rank(model, str(boxes_in))
        if r is not None and r >= 2:
            axis = int(r - 1)
        swapped_boxes = str(boxes_in) + "_xyxy"
        new_nodes.append(
            helper.make_node(
                "Gather",
                [boxes_in, swap_idx_name],
                [swapped_boxes],
                name=old.name + "_SwapXY",
                axis=int(axis),
            )
        )

    attr_payload = (
        "["
        f'{{"name":"max_output_boxes_per_class","i":{max_output}}},'
        f'{{"name":"iou_threshold","f":{iou_thr}}},'
        f'{{"name":"score_threshold","f":{score_thr}}}'
        "]"
    )
    new = helper.make_node(
        "NPUNonMaxSuppressionOrt",
        [swapped_boxes, scores_in],
        [out_name],
        name=old.name,
        max_output_boxes_per_class=max_output,
        iou_threshold=iou_thr,
        score_threshold=score_thr,
        attribute=attr_payload,
    )

    if new_nodes:
        nodes = list(model.graph.node)
        nodes.insert(nms_idx, new_nodes[0])
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        nms_idx += 1
    model.graph.node[nms_idx].CopyFrom(new)

    _dedupe_opset_import(model)
    _prune_unused_opset_import(model)
    return {"status": "ok", "max_output": max_output, "iou": iou_thr, "score": score_thr}


def _has_init(m: onnx.ModelProto, name: str) -> bool:
    return any(i.name == name for i in m.graph.initializer)


def _add_init_i64(m: onnx.ModelProto, name: str, vals: list[int]) -> None:
    if _has_init(m, name):
        return
    arr = np.asarray(vals, dtype=np.int64)
    t = helper.make_tensor(name=name, data_type=TensorProto.INT64, dims=[int(arr.size)], vals=arr)
    m.graph.initializer.append(t)


def _patch_fullgraph_nms_squeeze_to_reshape(model: onnx.ModelProto) -> dict:
    target_name = "/Squeeze"
    target_out = "/Squeeze_output_0"

    idx = -1
    for i, n in enumerate(model.graph.node):
        if n.name == target_name and n.op_type == "Squeeze" and list(n.output) == [target_out]:
            idx = i
            break
    if idx < 0:
        raise RuntimeError(f"target squeeze not found: name={target_name} out={target_out}")

    n = model.graph.node[idx]
    if len(n.input) != 1:
        raise RuntimeError(f"unexpected squeeze input count={len(n.input)} for {n.name}")
    x = n.input[0]

    const_axes0 = "__nms_sq2rs_axes0"
    const_steps1 = "__nms_sq2rs_steps1"
    const_start0 = "__nms_sq2rs_start0"
    const_end1 = "__nms_sq2rs_end1"
    _add_init_i64(model, const_axes0, [0])
    _add_init_i64(model, const_steps1, [1])
    _add_init_i64(model, const_start0, [0])
    _add_init_i64(model, const_end1, [1])

    shape_out = f"{target_out}__shape"
    slice_out = f"{target_out}__shape_sliced"
    shape_node = helper.make_node("Shape", [x], [shape_out], name=f"{target_name}__Shape")
    slice_node = helper.make_node(
        "Slice",
        [shape_out, const_start0, const_end1, const_axes0, const_steps1],
        [slice_out],
        name=f"{target_name}__SliceShape",
    )

    model.graph.node.insert(idx, shape_node)
    model.graph.node.insert(idx + 1, slice_node)

    n.op_type = "Reshape"
    n.domain = ""
    del n.attribute[:]
    del n.input[:]
    n.input.extend([x, slice_out])
    return {"patched": [target_name]}


def _patch_atc_ready_extras(model: onnx.ModelProto) -> dict:
    need: dict[str, tuple[int, list[int]]] = {
        "/Unsqueeze_75_output_0": (TensorProto.FLOAT, [1, 2000, 1, 5]),
        "/Slice_21_output_0": (TensorProto.FLOAT, [1, 2000, 15]),
    }
    vi_map = {v.name: v for v in model.graph.value_info}
    changed = False
    for name, (dtype, shape) in need.items():
        vi = vi_map.get(name)
        if vi is None:
            model.graph.value_info.extend([helper.make_tensor_value_info(name, dtype, shape)])
            changed = True
            continue
        tt = vi.type.tensor_type
        if int(tt.elem_type) == 0:
            tt.elem_type = int(dtype)
            changed = True
        if not tt.HasField("shape") or len(tt.shape.dim) != len(shape):
            vi.CopyFrom(helper.make_tensor_value_info(name, dtype, shape))
            changed = True
            continue
        for i, d in enumerate(shape):
            dim = tt.shape.dim[i]
            if (not dim.HasField("dim_value")) or int(dim.dim_value) != int(d):
                dim.dim_value = int(d)
                if dim.HasField("dim_param"):
                    dim.ClearField("dim_param")
                changed = True

    init_names = {i.name for i in model.graph.initializer if i.name}
    tensor_names = set()
    for n in model.graph.node:
        tensor_names.update([x for x in n.input if x])
        tensor_names.update([x for x in n.output if x])
    for i in model.graph.input:
        tensor_names.add(i.name)
    for o in model.graph.output:
        tensor_names.add(o.name)
    for v in model.graph.value_info:
        tensor_names.add(v.name)

    def uniq(base: str) -> str:
        if base not in tensor_names and base not in init_names:
            tensor_names.add(base)
            return base
        k = 0
        while True:
            cand = f"{base}_{k}"
            if cand not in tensor_names and cand not in init_names:
                tensor_names.add(cand)
                return cand
            k += 1

    target_loc_name = "/Unsqueeze_75_output_0"
    target_conf_name = "/Slice_21_output_0"
    loc_shape = np.array([1, 2000, 1, 5], dtype=np.int64)
    conf_shape = np.array([1, 2000, 15], dtype=np.int64)

    loc_shape_init = uniq("fixshape_loc_shape")
    conf_shape_init = uniq("fixshape_conf_shape")
    if loc_shape_init not in init_names:
        model.graph.initializer.extend([onnx.numpy_helper.from_array(loc_shape, name=loc_shape_init)])
        init_names.add(loc_shape_init)
        changed = True
    if conf_shape_init not in init_names:
        model.graph.initializer.extend([onnx.numpy_helper.from_array(conf_shape, name=conf_shape_init)])
        init_names.add(conf_shape_init)
        changed = True

    new_nodes: list[onnx.NodeProto] = []
    inserted = False
    for n in model.graph.node:
        if n.op_type == "NPUBatchedRotatedNMS" and len(n.input) >= 2:
            if n.input[0] == target_loc_name:
                fixed_loc = uniq(target_loc_name + "__fixedshape")
                new_nodes.append(helper.make_node("Reshape", [target_loc_name, loc_shape_init], [fixed_loc], name=uniq("FixShape_Loc")))
                n.input[0] = fixed_loc
                inserted = True
            if n.input[1] == target_conf_name:
                fixed_conf = uniq(target_conf_name + "__fixedshape")
                new_nodes.append(helper.make_node("Reshape", [target_conf_name, conf_shape_init], [fixed_conf], name=uniq("FixShape_Conf")))
                n.input[1] = fixed_conf
                inserted = True
        new_nodes.append(n)
    if inserted:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        changed = True

    init_map = {i.name: i for i in model.graph.initializer if i.name}
    if "2009" in init_map:
        arr = onnx.numpy_helper.to_array(init_map["2009"])
        if arr.shape == (3,) and int(arr[0]) == 1 and int(arr[2]) == 5 and int(arr[1]) == -1:
            arr2 = np.array([1, 2000, 5], dtype=np.int64)
            init_map["2009"].CopyFrom(onnx.numpy_helper.from_array(arr2, name="2009"))
            changed = True

    _keep_single_default_domain_opset(model)
    return {"value_info_patched": bool(changed), "reshape_inserted": bool(inserted)}


def _patch_end2end_simp_ops_to_npu(model: onnx.ModelProto, enable_npu_gelu: bool = False) -> tuple[onnx.ModelProto, int]:
    if not hasattr(onnx, "mapping"):
        try:
            import types

            tmap = getattr(getattr(onnx, "_mapping", None), "TENSOR_TYPE_MAP", None)
            if isinstance(tmap, dict) and tmap:
                t2n = {k: v.np_dtype for k, v in tmap.items() if hasattr(v, "np_dtype")}
                n2t = {np.dtype(v): k for k, v in t2n.items()}
                onnx.mapping = types.SimpleNamespace(TENSOR_TYPE_TO_NP_TYPE=t2n, NP_TYPE_TO_TENSOR_TYPE=n2t)
        except Exception:
            pass

    try:
        import onnx_graphsurgeon as gs
    except Exception as e:
        raise RuntimeError(
            "onnx_graphsurgeon 未安装。可按 NVIDIA 文档安装：\n"
            "python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com\n"
            f"原始错误: {e}"
        )

    def _prune_opset_import(model2: onnx.ModelProto) -> None:
        used_domains = {str(getattr(n, "domain", "") or "") for n in model2.graph.node}
        keep = []
        for o in model2.opset_import:
            if o.domain in used_domains:
                keep.append((o.domain, int(o.version)))
            elif o.domain == "" and "" in used_domains:
                keep.append((o.domain, int(o.version)))
            elif o.domain == "ai.onnx" and "ai.onnx" in used_domains:
                keep.append((o.domain, int(o.version)))
        del model2.opset_import[:]
        for domain, ver in keep:
            op = model2.opset_import.add()
            op.domain = domain
            op.version = ver

    mapping = {
        "MMCVMultiLevelRotatedRoiAlign": "NPUMMCVMultiLevelRotatedRoiAlign",
        "MMCVMultiLevelRoiAlign": "NPUMMCVMultiLevelRoiAlign",
        "TRTBatchedRotatedNMS": "NPUBatchedRotatedNMS",
        "GridPriorsTRT": "GridPriorsNPU",
    }
    npu_ops = set(mapping.values())
    if enable_npu_gelu:
        npu_ops.add("NPUGelu")

    graph = gs.import_onnx(model)
    changed = 0
    for node in graph.nodes:
        before = (node.op, node.domain)
        if node.op in mapping:
            node.op = mapping[node.op]
        if node.op in npu_ops:
            node.domain = ""
        after = (node.op, node.domain)
        if after != before:
            changed += 1

    def _try_fuse_layernorm(add1_node: "gs.Node") -> "tuple[gs.Node, set[int]] | None":
        if add1_node.op != "Add" or len(add1_node.inputs) != 2 or len(add1_node.outputs) != 1:
            return None

        in0, in1 = add1_node.inputs
        if isinstance(in0, gs.Constant) ^ isinstance(in1, gs.Constant):
            beta = in0 if isinstance(in0, gs.Constant) else in1
            mul_out = in1 if isinstance(in0, gs.Constant) else in0
        else:
            return None

        if len(getattr(mul_out, "inputs", [])) != 1:
            return None
        mul = mul_out.inputs[0]
        if mul.op != "Mul" or len(mul.inputs) != 2:
            return None
        m0, m1 = mul.inputs
        if isinstance(m0, gs.Constant) ^ isinstance(m1, gs.Constant):
            gamma = m0 if isinstance(m0, gs.Constant) else m1
            div_out = m1 if isinstance(m0, gs.Constant) else m0
        else:
            return None

        if len(getattr(div_out, "inputs", [])) != 1:
            return None
        div = div_out.inputs[0]
        if div.op != "Div" or len(div.inputs) != 2:
            return None
        sub_out, sqrt_out = div.inputs
        if len(getattr(sub_out, "inputs", [])) != 1 or len(getattr(sqrt_out, "inputs", [])) != 1:
            return None
        sub = sub_out.inputs[0]
        sqrt = sqrt_out.inputs[0]
        if sub.op != "Sub" or sqrt.op != "Sqrt":
            return None

        if len(sub.inputs) != 2:
            return None
        x, mean_out = sub.inputs
        if len(getattr(mean_out, "inputs", [])) != 1:
            return None
        mean = mean_out.inputs[0]
        if mean.op != "ReduceMean":
            return None
        if mean.attrs.get("keepdims", 1) != 1 or mean.attrs.get("axes", None) not in ([-1], (-1,)):
            return None

        if len(getattr(sqrt.inputs[0], "inputs", [])) != 1:
            return None
        add_eps = sqrt.inputs[0].inputs[0]
        if add_eps.op != "Add" or len(add_eps.inputs) != 2:
            return None
        a0, a1 = add_eps.inputs
        if isinstance(a0, gs.Constant) ^ isinstance(a1, gs.Constant):
            eps_c = a0 if isinstance(a0, gs.Constant) else a1
            var_out = a1 if isinstance(a0, gs.Constant) else a0
        else:
            return None
        try:
            eps_val = float(np.array(eps_c.values).reshape(-1)[0])
        except Exception:
            return None

        if len(getattr(var_out, "inputs", [])) != 1:
            return None
        var = var_out.inputs[0]
        if var.op != "ReduceMean" or var.attrs.get("keepdims", 1) != 1 or var.attrs.get("axes", None) not in ([-1], (-1,)):
            return None

        if len(var.inputs) != 1:
            return None
        pow_out = var.inputs[0]
        if len(getattr(pow_out, "inputs", [])) != 1:
            return None
        pow_node = pow_out.inputs[0]
        if pow_node.op != "Pow" or len(pow_node.inputs) != 2:
            return None
        if pow_node.inputs[0] is not sub_out:
            return None
        if not isinstance(pow_node.inputs[1], gs.Constant):
            return None
        try:
            p = float(np.array(pow_node.inputs[1].values).reshape(-1)[0])
        except Exception:
            return None
        if abs(p - 2.0) > 1e-6:
            return None

        if not isinstance(gamma, gs.Constant) or not isinstance(beta, gs.Constant):
            return None
        if np.array(gamma.values).ndim != 1:
            return None
        hidden = int(np.array(gamma.values).shape[0])
        norm_shape = gs.Constant(name=f"{add1_node.name}_normalized_shape", values=np.array([hidden], dtype=np.int32))

        y = add1_node.outputs[0]
        if hasattr(y, "inputs"):
            y.inputs.clear()
        ln = gs.Node(
            op="NPULayerNorm",
            name=f"{add1_node.name}_NPULayerNorm",
            inputs=[x, norm_shape, gamma, beta],
            outputs=[y],
            attrs={"epsilon": eps_val},
        )
        ln.domain = ""

        old_nodes = [add1_node, mul, div, sub, sqrt, add_eps, var, pow_node, mean]
        old_node_ids = {id(n) for n in old_nodes}
        for t in [sub_out, sqrt_out, mul_out, mean_out, var_out, pow_out]:
            for n in getattr(t, "outputs", []):
                if id(n) not in old_node_ids:
                    return None
        return ln, set(old_node_ids)

    fused = 0
    new_nodes = []
    remove_nodes: set[int] = set()
    for node in list(graph.nodes):
        if node.op == "Add" and node.name and "/norm" in node.name and node.name.endswith("/Add_1"):
            res = _try_fuse_layernorm(node)
            if res is not None:
                ln, old_node_ids = res
                new_nodes.append(ln)
                remove_nodes.update(old_node_ids)
                fused += 1
    if fused:
        graph.nodes.extend(new_nodes)
        graph.nodes = [n for n in graph.nodes if id(n) not in remove_nodes]
        graph.cleanup().toposort()
        changed += fused

    def _try_fuse_gelu(mul2_node: "gs.Node") -> "tuple[gs.Node, set[int]] | None":
        def _const_scalar(v: "gs.Variable | gs.Constant") -> "float | None":
            if isinstance(v, gs.Constant):
                try:
                    return float(np.array(v.values).reshape(-1)[0])
                except Exception:
                    return None
            if hasattr(v, "inputs") and len(getattr(v, "inputs", [])) == 1:
                n0 = v.inputs[0]
                if n0.op == "Constant":
                    val = n0.attrs.get("value", None)
                    if val is None:
                        return None
                    try:
                        return float(np.array(val).reshape(-1)[0])
                    except Exception:
                        return None
            return None

        if mul2_node.op != "Mul" or len(mul2_node.inputs) != 2 or len(mul2_node.outputs) != 1:
            return None
        a0, a1 = mul2_node.inputs
        half_v0 = _const_scalar(a0)
        half_v1 = _const_scalar(a1)
        if (half_v0 is None) == (half_v1 is None):
            return None
        half_v = half_v0 if half_v0 is not None else half_v1
        mul1_out = a1 if half_v0 is not None else a0
        if abs(half_v - 0.5) > 1e-4:
            return None
        if len(getattr(mul1_out, "inputs", [])) != 1:
            return None
        mul1 = mul1_out.inputs[0]
        if mul1.op != "Mul" or len(mul1.inputs) != 2:
            return None
        m0, m1 = mul1.inputs

        def _match_add_one_erf(v: "gs.Variable") -> "tuple[gs.Node, gs.Variable] | None":
            if len(getattr(v, "inputs", [])) != 1:
                return None
            add_node = v.inputs[0]
            if add_node.op != "Add" or len(add_node.inputs) != 2 or len(add_node.outputs) != 1:
                return None
            b0, b1 = add_node.inputs
            one_v0 = _const_scalar(b0)
            one_v1 = _const_scalar(b1)
            if (one_v0 is None) == (one_v1 is None):
                return None
            one_v = one_v0 if one_v0 is not None else one_v1
            if abs(one_v - 1.0) > 1e-4:
                return None
            erf_var = b1 if one_v0 is not None else b0
            if len(getattr(erf_var, "inputs", [])) != 1:
                return None
            erf_node = erf_var.inputs[0]
            if erf_node.op != "Erf" or len(erf_node.inputs) != 1:
                return None
            return add_node, erf_var

        cand0 = _match_add_one_erf(m0) if hasattr(m0, "inputs") else None
        cand1 = _match_add_one_erf(m1) if hasattr(m1, "inputs") else None
        if (cand0 is None) == (cand1 is None):
            return None
        add = cand0[0] if cand0 is not None else cand1[0]
        erf_out = cand0[1] if cand0 is not None else cand1[1]
        add_out = m0 if cand0 is not None else m1
        x = m1 if cand0 is not None else m0
        if len(getattr(erf_out, "inputs", [])) != 1:
            return None
        erf = erf_out.inputs[0]
        if erf.op != "Erf" or len(erf.inputs) != 1:
            return None
        div_out = erf.inputs[0]
        if len(getattr(div_out, "inputs", [])) != 1:
            return None
        div = div_out.inputs[0]
        if div.op != "Div" or len(div.inputs) != 2:
            return None
        d0, d1 = div.inputs
        c_v0 = _const_scalar(d0)
        c_v1 = _const_scalar(d1)
        if (c_v0 is None) == (c_v1 is None):
            return None
        c_v = c_v0 if c_v0 is not None else c_v1
        x2 = d1 if c_v0 is not None else d0
        if x2 is not x:
            return None
        if abs(c_v - 1.41421356237) > 5e-3 and abs(c_v - 1.4142) > 5e-3:
            return None

        y = mul2_node.outputs[0]
        if hasattr(y, "inputs"):
            y.inputs.clear()
        gelu = gs.Node(op="NPUGelu", name=f"{mul2_node.name}_NPUGelu", inputs=[x], outputs=[y])
        gelu.domain = ""

        old_nodes = [mul2_node, mul1, add, erf, div]
        old_ids = {id(n) for n in old_nodes}
        for t in [mul1_out, add_out, erf_out, div_out]:
            for n in getattr(t, "outputs", []):
                if id(n) not in old_ids:
                    return None
        return gelu, old_ids

    gelu_fused = 0
    if enable_npu_gelu:
        gelu_new = []
        gelu_rm: set[int] = set()
        for node in list(graph.nodes):
            if node.op == "Mul":
                res = _try_fuse_gelu(node)
                if res is not None:
                    g, old_ids = res
                    gelu_new.append(g)
                    gelu_rm.update(old_ids)
                    gelu_fused += 1
        if gelu_fused:
            graph.nodes.extend(gelu_new)
            graph.nodes = [n for n in graph.nodes if id(n) not in gelu_rm]
            graph.cleanup().toposort()
            changed += gelu_fused

    out_model = gs.export_onnx(graph)
    _prune_opset_import(out_model)
    for n in out_model.graph.node:
        if n.op_type in npu_ops:
            _add_attribute_string_for_atc(n)
    _patch_grid_priors_hw_inputs(out_model)
    return out_model, changed


def _decompose_npu_layernorm(model: onnx.ModelProto) -> int:
    init_names = {i.name for i in model.graph.initializer if i.name}

    def uniq(base: str) -> str:
        if base not in init_names:
            return base
        idx = 0
        while True:
            cand = f"{base}_{idx}"
            if cand not in init_names:
                return cand
            idx += 1

    def add_scalar_init(base: str, value: float) -> str:
        name = uniq(base)
        arr = np.asarray(value, dtype=np.float32)
        model.graph.initializer.append(onnx.numpy_helper.from_array(arr, name=name))
        init_names.add(name)
        return name

    new_nodes: list[onnx.NodeProto] = []
    changed = 0
    for idx, n in enumerate(model.graph.node):
        if n.op_type != "NPULayerNorm":
            new_nodes.append(n)
            continue
        if len(n.input) < 4 or len(n.output) < 1:
            new_nodes.append(n)
            continue

        x, _norm_shape, gamma, beta = n.input[:4]
        y_out = n.output[0]
        mean_out = n.output[1] if len(n.output) > 1 and n.output[1] else f"{n.name}_mean"
        rstd_out = n.output[2] if len(n.output) > 2 and n.output[2] else f"{n.name}_rstd"
        eps = 1e-5
        for a in n.attribute:
            if a.name == "epsilon":
                eps = float(a.f)
                break

        base = n.name or f"NPULayerNorm_{idx}"
        sub_out = f"{base}__sub"
        pow_out = f"{base}__pow"
        var_out = f"{base}__var"
        var_eps_out = f"{base}__var_eps"
        std_out = f"{base}__std"
        norm_out = f"{base}__norm"
        scaled_out = f"{base}__scaled"

        two_c = add_scalar_init(f"{base}__const_two", 2.0)
        eps_c = add_scalar_init(f"{base}__const_eps", float(eps))
        one_c = add_scalar_init(f"{base}__const_one", 1.0)

        new_nodes.extend(
            [
                helper.make_node("ReduceMean", [x], [mean_out], name=f"{base}__ReduceMean0", axes=[-1], keepdims=1),
                helper.make_node("Sub", [x, mean_out], [sub_out], name=f"{base}__Sub"),
                helper.make_node("Pow", [sub_out, two_c], [pow_out], name=f"{base}__Pow"),
                helper.make_node("ReduceMean", [pow_out], [var_out], name=f"{base}__ReduceMean1", axes=[-1], keepdims=1),
                helper.make_node("Add", [var_out, eps_c], [var_eps_out], name=f"{base}__AddEps"),
                helper.make_node("Sqrt", [var_eps_out], [std_out], name=f"{base}__Sqrt"),
                helper.make_node("Div", [one_c, std_out], [rstd_out], name=f"{base}__RStd"),
                helper.make_node("Mul", [sub_out, rstd_out], [norm_out], name=f"{base}__Norm"),
                helper.make_node("Mul", [norm_out, gamma], [scaled_out], name=f"{base}__Scale"),
                helper.make_node("Add", [scaled_out, beta], [y_out], name=f"{base}__Bias"),
            ]
        )
        changed += 1

    if changed:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return changed


def _decompose_npu_gelu(model: onnx.ModelProto) -> int:
    init_names = {i.name for i in model.graph.initializer if i.name}

    def uniq(base: str) -> str:
        if base not in init_names:
            return base
        idx = 0
        while True:
            cand = f"{base}_{idx}"
            if cand not in init_names:
                return cand
            idx += 1

    def add_scalar_init(base: str, value: float) -> str:
        name = uniq(base)
        arr = np.asarray(value, dtype=np.float32)
        model.graph.initializer.append(onnx.numpy_helper.from_array(arr, name=name))
        init_names.add(name)
        return name

    new_nodes: list[onnx.NodeProto] = []
    changed = 0
    for idx, n in enumerate(model.graph.node):
        if n.op_type != "NPUGelu":
            new_nodes.append(n)
            continue
        if len(n.input) < 1 or len(n.output) < 1:
            new_nodes.append(n)
            continue

        x = n.input[0]
        y = n.output[0]
        base = n.name or f"NPUGelu_{idx}"

        half_c = add_scalar_init(f"{base}__const_half", 0.5)
        one_c = add_scalar_init(f"{base}__const_one", 1.0)
        sqrt2_c = add_scalar_init(f"{base}__const_sqrt2", 1.41421356237)

        div_out = f"{base}__div"
        erf_out = f"{base}__erf"
        add_out = f"{base}__add"
        mul1_out = f"{base}__mul1"

        new_nodes.extend(
            [
                helper.make_node("Div", [x, sqrt2_c], [div_out], name=f"{base}__Div"),
                helper.make_node("Erf", [div_out], [erf_out], name=f"{base}__Erf"),
                helper.make_node("Add", [erf_out, one_c], [add_out], name=f"{base}__Add"),
                helper.make_node("Mul", [x, add_out], [mul1_out], name=f"{base}__Mul0"),
                helper.make_node("Mul", [mul1_out, half_c], [y], name=f"{base}__Mul1"),
            ]
        )
        changed += 1

    if changed:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return changed


def patch_end2end_simp_atc_ready_base(
    in_onnx: Path,
    out_onnx: Path,
    enable_npu_gelu: bool = False,
    decompose_npu_layernorm: bool = False,
) -> dict:
    model = onnx.load(str(in_onnx))
    model, changed = _patch_end2end_simp_ops_to_npu(model, enable_npu_gelu=enable_npu_gelu)
    gelu_decomp = _decompose_npu_gelu(model)
    ln_decomp = _decompose_npu_layernorm(model) if decompose_npu_layernorm else 0
    patched_reduce = patch_reduce_axes_for_atc(model)
    extra = _patch_atc_ready_extras(model)
    _dedupe_opset_import(model)
    _prune_unused_opset_import(model)
    _keep_single_default_domain_opset(model)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_onnx))
    return {
        "ops_to_npu_changed": int(changed),
        "npugelu_decomposed": int(gelu_decomp),
        "npulayernorm_decomposed": int(ln_decomp),
        "reduce_axes_patched": int(patched_reduce),
        **extra,
        "out": str(out_onnx),
    }


def patch_end2end_simp_atc_ready_full(
    in_onnx: Path,
    out_onnx: Path,
    enable_npu_gelu: bool = False,
    decompose_npu_layernorm: bool = False,
) -> dict:
    tmp = out_onnx.with_suffix(".atc_ready_base.onnx")
    base_meta = patch_end2end_simp_atc_ready_base(
        in_onnx,
        tmp,
        enable_npu_gelu=enable_npu_gelu,
        decompose_npu_layernorm=decompose_npu_layernorm,
    )
    model = onnx.load(str(tmp))
    topk_meta = _patch_topk_to_ascend_topk(model, opset=11)
    nms_meta = _patch_nonmaxsuppression_to_npu_nms_ort(model)
    sq_meta = _patch_fullgraph_nms_squeeze_to_reshape(model)
    _keep_single_default_domain_opset(model)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_onnx))
    try:
        tmp.unlink()
    except Exception:
        pass
    return {"base": base_meta, "topk": topk_meta, "nms": nms_meta, "sq": sq_meta, "out": str(out_onnx)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pipeline",
        required=True,
        choices=[
            "ops_to_npu",
            "decompose_npu_layernorm",
            "reduce_axes",
            "atc_ready_extras",
            "topk_to_ascend_topk",
            "nms_to_npu_nms_ort",
            "nms_squeeze_to_reshape",
            "atc_ready_base",
            "atc_ready_full",
        ],
    )
    p.add_argument("--in-onnx", required=True)
    p.add_argument("--out-onnx", required=True)
    p.add_argument("--opset", type=int, default=11)
    p.add_argument("--nms-node-name", default="/NonMaxSuppression")
    p.add_argument("--swap-boxes-xy", action="store_true")
    p.add_argument("--enable-npu-gelu", action="store_true")
    p.add_argument("--decompose-npu-layernorm", action="store_true")
    args = p.parse_args()

    inp = Path(args.in_onnx)
    outp = Path(args.out_onnx)
    if not inp.is_file():
        raise SystemExit(f"missing --in-onnx: {inp}")

    if args.pipeline == "ops_to_npu":
        model = onnx.load(str(inp))
        model, changed = _patch_end2end_simp_ops_to_npu(model, enable_npu_gelu=bool(args.enable_npu_gelu))
        outp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(outp))
        meta = {"ops_to_npu_changed": int(changed), "out": str(outp)}
    elif args.pipeline == "reduce_axes":
        model = onnx.load(str(inp))
        patched = patch_reduce_axes_for_atc(model)
        outp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(outp))
        meta = {"reduce_axes_patched": int(patched), "out": str(outp)}
    elif args.pipeline == "decompose_npu_layernorm":
        model = onnx.load(str(inp))
        patched = _decompose_npu_layernorm(model)
        outp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(outp))
        meta = {"npulayernorm_decomposed": int(patched), "out": str(outp)}
    elif args.pipeline == "atc_ready_extras":
        model = onnx.load(str(inp))
        meta = _patch_atc_ready_extras(model)
        outp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(outp))
        meta["out"] = str(outp)
    elif args.pipeline == "topk_to_ascend_topk":
        model = onnx.load(str(inp))
        meta = _patch_topk_to_ascend_topk(model, opset=int(args.opset))
        outp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(outp))
        meta["out"] = str(outp)
    elif args.pipeline == "nms_to_npu_nms_ort":
        model = onnx.load(str(inp))
        meta = _patch_nonmaxsuppression_to_npu_nms_ort(
            model,
            nms_node_name=str(args.nms_node_name),
            swap_boxes_xy=bool(args.swap_boxes_xy),
        )
        outp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(outp))
        meta["out"] = str(outp)
    elif args.pipeline == "nms_squeeze_to_reshape":
        model = onnx.load(str(inp))
        meta = _patch_fullgraph_nms_squeeze_to_reshape(model)
        outp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(outp))
        meta["out"] = str(outp)
    elif args.pipeline == "atc_ready_base":
        meta = patch_end2end_simp_atc_ready_base(
            inp,
            outp,
            enable_npu_gelu=bool(args.enable_npu_gelu),
            decompose_npu_layernorm=bool(args.decompose_npu_layernorm),
        )
    else:
        meta = patch_end2end_simp_atc_ready_full(
            inp,
            outp,
            enable_npu_gelu=bool(args.enable_npu_gelu),
            decompose_npu_layernorm=bool(args.decompose_npu_layernorm),
        )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

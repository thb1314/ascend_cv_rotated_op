# Open Source Release (No Operator Source)

本包用于开源 **从 `end2end_simp.onnx` 到 `OM` 推理与可视化** 的完整流程。  
不包含 `op_host/op_kernel/framework` 全量算子源码，但保留了 **算子替换相关源码**（`Where_2`、`Gather_40` 改写脚本）用于复现图改写与精度对齐。

## 1. 包含与不包含

- 包含：流程脚本、ONNX 模型、ONNXRuntime 运行库、ORT 自定义算子 `.so`、示例图片、各 SOC 的 `.run` 安装包。
- 不包含：`ascend_c_onnx` 下各算子完整源码工程。
- 说明：`Ascend310P`（无后缀）在 910 构建环境失败，因此仅提供 `Ascend310P1/P3`。

## 2. 运行环境依赖（Conda Base）

### 2.1 版本依赖（CANN / 驱动 / 固件）

以下为本仓库在当前设备上的**已验证组合**（Ascend310B4 / Atlas 200I A2）：

- CANN（ATC/Compiler）：`8.5.0`
  - 参考：`/usr/local/Ascend/cann-8.5.0`、`/usr/local/Ascend/cann-8.5.0/compiler/version.info`
- Driver / npu-smi：`25.5.0`
  - 参考：`/var/davinci/driver/version.info`、`npu-smi info`
- Board Firmware：`6.2.2.0.b133`
  - 参考：`npu-smi info -t board -i 0`

建议：
- 目标机器尽量使用与上述一致的 CANN + Driver + Firmware 组合。
- 至少保证 `npu-smi info -t board -i 0` 中 `Software Version` 与 `Firmware Version` 正常可读，再执行 ATC 和 OM 推理。

版本自检命令：

```bash
cat /usr/local/Ascend/cann-8.5.0/compiler/version.info
cat /var/davinci/driver/version.info
npu-smi info -t board -i 0
```

### 2.2 系统依赖

- CANN/ATC：已安装并可用（示例路径：`/usr/local/Ascend/cann-8.5.0/bin/atc`）。
- Ascend 运行环境：`/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash` 可 source。
- 常用工具：`bash`、`screen`、`sha256sum`。

### 2.3 Python 依赖（在 `conda base`）

```bash

conda activate base

python -m pip install -U numpy onnx onnxruntime opencv-python
python -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
```

说明：
- 本仓库脚本默认使用 `conda base`（例如 `run_verify_310b4.sh`、workflow 脚本）。
- 若目标机器没有 conda，先安装 Miniconda，再按上面命令安装依赖。

## 3. 算子包安装方式（必须）

以下以 `Ascend310B4` 为例，其他 SOC 只需替换 `SOC_VERSION`：

```bash
REPO_ROOT=/path/to/open_source_release_20260222
cd "$REPO_ROOT/standalone_end2end_simp_runner"

SOC_VERSION=Ascend310B4
INSTALL_BASE="$PWD/.ascend_custom_opp_mmrotate_all_ops"
RUN_PKG="$PWD/opp_pkgs/mmrotate_all_ops_frameworklaunch_${SOC_VERSION}_custom_opp_ubuntu_aarch64.run"

# 可选：先校验安装包
sha256sum -c "$PWD/opp_pkgs/SHA256SUMS.txt" --ignore-missing

# 重装（确保当前测试使用本包内算子）
rm -rf "$INSTALL_BASE"
"$RUN_PKG" --quiet --install-path="$INSTALL_BASE"

export ASCEND_CUSTOM_OPP_PATH="$INSTALL_BASE/vendors/customize"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$ASCEND_CUSTOM_OPP_PATH/framework/onnx:$PWD/onnxruntime-linux-aarch64-1.11.0/lib:${LD_LIBRARY_PATH:-}"
```

可用 SOC 包位于 `standalone_end2end_simp_runner/opp_pkgs/`：
- `Ascend310B1/B2/B3/B4`
- `Ascend310P1/P3`
- `Ascend910B/910B1/910B2/910B3/910B4`
- `Ascend910_93`

## 4. 快速开始（一键）

```bash
REPO_ROOT=/path/to/open_source_release_20260222
cd "$REPO_ROOT"

conda activate base

# 重新安装本包算子 + ONNX 替换 + screen+ATC + OM/ORT 对比 + 可视化
bash run_verify_310b4.sh
```

## 5. 标准流程（ONNX -> OM -> 对比）

1. 生成 ATC 友好 ONNX（含必要替换）  
`standalone_end2end_simp_runner/workflows/end2end_fullgraph_om_vis/prepare_fullgraph_onnx.py`

2. 用 `screen` 后台转 OM  
`standalone_end2end_simp_runner/workflows/end2end_fullgraph_om_vis/build_full_om_in_screen.sh`  
`standalone_end2end_simp_runner/tools/run_atc_in_screen.sh`

3. 跑 OM、补跑/复用 ORT、产出指标与可视化  
`standalone_end2end_simp_runner/workflows/end2end_fullgraph_om_vis/run_compare_and_visualize.sh`  
`standalone_end2end_simp_runner/scripts/fill_ort_vis_and_benchmark.py`

脚本参数总览见：
- `standalone_end2end_simp_runner/README_SCRIPT_ARGS.md`

## 6. 自定义算子说明（含输入输出 Shape）

以下 shape 为当前流程实测（固定输入 `input: [1, 3, 672, 1024]`）。

### 6.1 `NPUMMCVMultiLevelRoiAlign`
- 功能：多层 FPN 的轴对齐 RoIAlign，输出固定大小 RoI 特征。
- 输入：
  - `rois`: `[2000, 5]`（`[batch_id, x1, y1, x2, y2]`）
  - `feat0`: `[1, 256, 168, 256]`
  - `feat1`: `[1, 256, 84, 128]`
  - `feat2`: `[1, 256, 42, 64]`
  - `feat3`: `[1, 256, 21, 32]`
- 输出：
  - `y`: `[2000, 256, 7, 7]`
- 关键属性：`output_height/output_width`、`sampling_ratio`、`aligned`、`finest_scale`、`featmap_stride0~3`。

### 6.2 `NPUMMCVMultiLevelRotatedRoiAlign`
- 功能：多层 FPN 的旋转框 RoIAlign（含角度）。
- 输入：
  - `rois`: `[2000, 6]`（常见为 `[batch_id, cx, cy, w, h, angle]`）
  - `feat0..feat3`: 同 6.1
- 输出：
  - `y`: `[2000, 256, 7, 7]`
- 关键属性：`clockwise`、`aligned`、`sampling_ratio`、`roi_scale_factor`、`finest_scale`、`featmap_stride0~3`。

### 6.3 `GridPriorsNPU`
- 功能：按特征图尺度生成先验框网格（priors/anchors）。
- 输入（单个 level 示例）：
  - `base_anchor`: `[3, 4]`
  - `feat_h`: `[168]`（不同层可为 `[84]`/`[42]`/`[21]`）
  - `feat_w`: `[256]`（不同层可为 `[128]`/`[64]`/`[32]`）
- 输出：
  - `priors`: `[H*W*3, 4]`  
    例如 level0：`[129024, 4]`；其它层典型：`[32256, 4]`、`[8064, 4]`、`[2016, 4]`。
- 关键属性：`stride_h/stride_w`（控制先验中心间距与尺度映射）。

### 6.4 `NPUBatchedRotatedNMS`
- 功能：批量旋转框 NMS，输出最终检测框与标签。
- 输入：
  - `loc`: `[1, 2000, 1, 5]`
  - `conf`: `[1, 2000, 15]`
- 输出：
  - `dets`: `[1, 1000, 6]`（常见为框参数 + score）
  - `labels`: `[1, 1000]`
- 关键属性：`score_threshold`、`iou_threshold`、`topk`、`keep_topk`、`num_classes`。

### 6.5 `AscendTopK`
- 功能：候选分数 Top-K 预筛选。
- 输入（不同分支会出现不同长度）：
  - `x`: `[8064]` / `[32256]` / `[129024]`
- 输出：
  - `values`: `[3840]`
  - `indices`: `[3840]`
- 关键属性：`k`、`largest`、`sorted`。

### 6.6 `NonMaxSuppressionOrt`
- 功能：与 ONNX NMS 语义对齐的自定义 NMS。
- 输入：
  - `boxes`: `[1, 17376, 4]`
  - `scores`: `[1, 1, 17376]`
- 输出：
  - `selected_indices`: `[3000, 3]`（`[batch_id, class_id, box_id]`）
- 关键属性：`max_output_boxes_per_class`、`iou_threshold`、`score_threshold`。

### 6.7 `NMSWithMaskCustom`
- 功能：生成 NMS 掩码输出（`uint8`），用于后续筛选逻辑。
- 输入：
  - `x`: `float32` ND（候选框相关输入）
  - `y`: `float32` ND（掩码模板/配套输入）
- 输出：
  - `z`: `uint8` ND，且 **shape 与 `y` 完全一致**（算子实现中 `z_shape = y_shape`）。
- 备注：该算子在当前主流程中通常不作为核心瓶颈节点，实际 shape 由调用子图决定。

## 7. 关键目录与核心文件（精简）

| 路径 | 作用 |
|---|---|
| `run_verify_310b4.sh` | 一键验证入口：安装 OPP、图改写、ATC、OM/ORT 对比、可视化。 |
| `standalone_end2end_simp_runner/opp_pkgs/` | 各 SOC 的自定义算子安装包。 |
| `standalone_end2end_simp_runner/ascend_c_onnx/` | ONNX 模型输入（原始与中间模型）。 |
| `standalone_end2end_simp_runner/workflows/end2end_fullgraph_om_vis/` | 全流程编排脚本与步骤说明。 |
| `standalone_end2end_simp_runner/README_SCRIPT_ARGS.md` | 各脚本参数说明（环境变量 + CLI 参数）。 |
| `standalone_end2end_simp_runner/tools/patch_where2_maskarith.py` | `/Where_2` 替换源码（mask 算术子图）。 |
| `standalone_end2end_simp_runner/tools/patch_gather40_to_slice_squeeze.py` | `/Gather_40` 替换源码（`Slice+Squeeze`）。 |
| `standalone_end2end_simp_runner/scripts/common_vis_utils.py` | 公共预处理与可视化函数（最小依赖实现）。 |
| `standalone_end2end_simp_runner/onnxruntime_mmrotate_ops/build/libonnxruntime_mmrotate_ops.so` | ORT 自定义算子插件。 |
| `standalone_end2end_simp_runner/onnxruntime-linux-aarch64-1.11.0/lib/` | ORT 运行库（可直接打包使用）。 |
| `standalone_end2end_simp_runner/images/` | 端到端测试图片。 |

## 8. 结果文件说明

- 端到端报告（误差、性能、路径）位于 workflow 输出目录下的 JSON。
- 每张图片会生成可视化对比图（`vis_side_by_side.jpg`）。
- 若仅重跑 OM，可复用已有 ORT 中间结果以缩短时间。

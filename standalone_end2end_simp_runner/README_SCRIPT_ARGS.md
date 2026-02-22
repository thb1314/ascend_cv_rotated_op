# 脚本参数说明（最小依赖版）

本文档对应当前精简后的脚本集合，目标是：
- 保留 `ONNX -> OM -> OM/ORT 对比可视化` 的最小可运行链路。
- 明确每个脚本的可配置参数（环境变量或 CLI 参数）。

约定：
- `REPO_ROOT`：`open_source_release_20260222/standalone_end2end_simp_runner`
- 默认输入尺寸：`input:1,3,672,1024`

## 1. 精简结果

已移除无用脚本：
- `scripts/batch_ort_vs_acl_end2end_simp.py`

原因：
- 不在主流程调用链中。
- 依赖未纳入发布包的模块（`scripts.debug_compare_bins_vs_om`）。

新增轻量公共模块：
- `scripts/common_vis_utils.py`
- 用于承载预处理与可视化函数，避免主流程脚本再依赖已移除脚本。

## 2. Shell 脚本参数

### 2.1 `run_verify_310b4.sh`

用途：一键验证入口（重装 OPP + prepare + screen+ATC + OM/ORT 对比）

可配置环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SOC_VERSION` | `Ascend310B4` | 目标 SOC。 |
| `PRECISION_MODE` | `allow_fp32_to_fp16` | ATC 精度模式。 |
| `LIMIT` | `10` | 参与对比的图片数量。 |

说明：
- 脚本内部固定 `REUSE_OM_IF_EXISTS=0`，每次都会重建 OM。

### 2.2 `env.sh`

用途：安装/加载自定义 OPP，设置运行时环境变量。

可配置环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SOC_VERSION` | `Ascend910B2` | 选择 `.run` 安装包名中的 SOC。 |
| `DEVICE_ID` | `0` | 设备 ID（本脚本内不直接使用，供调用方统一传递）。 |
| `MAX_COMPILE_CORE_NUMBER` | `1` | 编译并发核数限制。 |

脚本导出：
- `ASCEND_CUSTOM_OPP_PATH`
- `LD_LIBRARY_PATH`（追加自定义算子库与 ORT 库路径）

### 2.3 `tools/run_atc_in_screen.sh`

用途：以 `screen` 后台启动 ATC 命令并写日志。

可配置环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WORK_DIR` | 必填 | 工作目录（必填）。 |
| `CMD_SH` | `$WORK_DIR/atc_cmds.sh` | 需执行的 ATC 命令脚本。 |
| `SESSION_NAME` | `atc_<workdir>_<timestamp>` | screen 会话名。 |
| `LOG_FILE` | `$WORK_DIR/screen_<session>.log` | 日志文件路径。 |
| `ATC_TIMEOUT_SEC` | `1800` | 透传给 runner 的超时环境变量。 |
| `DEVICE_ID` | `0` | 设备 ID。 |

### 2.4 `workflows/end2end_fullgraph_om_vis/build_full_om_in_screen.sh`

用途：生成 ATC 命令并调用 `run_atc_in_screen.sh`。

可配置环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WORK_DIR` | 必填 | 工作目录。 |
| `MODEL_ONNX` | 必填 | 输入 ONNX 路径。 |
| `SOC_VERSION` | `Ascend310B4` | ATC 的 `--soc_version`。 |
| `OM_BASENAME` | `full_where2_gather40slice_allowfp32tofp16` | 输出 OM 前缀名。 |
| `PRECISION_MODE` | `allow_fp32_to_fp16` | ATC 精度模式。 |
| `DEVICE_ID` | `0` | 设备 ID。 |
| `ATC_EXTRA_ARGS` | 空 | 附加 ATC 参数原样拼接。 |
| `SESSION_NAME` | 自动生成 | 覆盖默认 screen 会话名。 |

### 2.5 `workflows/end2end_fullgraph_om_vis/run_compare_and_visualize.sh`

用途：执行 OM 推理、补跑 ORT、输出报告和可视化。

可配置环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OM_PATH` | 必填 | 输入 OM 路径。 |
| `OUT_DIR` | 必填 | 输出目录。 |
| `IMAGE_DIR` | `$REPO_ROOT/images` | 测试图片目录。 |
| `LIMIT` | `10` | 图片数量上限。 |
| `DEVICE_ID` | `0` | 设备 ID。 |
| `BENCH_WARMUP_OM` | `1` | 仅 OM benchmark 预热次数。 |
| `BENCH_ITERS_OM` | `3` | 仅 OM benchmark 迭代次数。 |
| `BENCH_WARMUP_BOTH` | `0` | ORT+OM benchmark 预热次数。 |
| `BENCH_ITERS_BOTH` | `1` | ORT+OM benchmark 迭代次数。 |
| `ORT_INTRA_OP_THREADS` | `8` | ORT intra-op 线程数。 |
| `ORT_INTER_OP_THREADS` | `1` | ORT inter-op 线程数。 |

### 2.6 `workflows/end2end_fullgraph_om_vis/run_all.sh`

用途：半自动流程（prepare + 启动 ATC，第三步手动执行）。

可配置环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `IN_ONNX` | `$REPO_ROOT/ascend_c_onnx/end2end_simp.onnx` | 输入 ONNX。 |
| `WORK_DIR` | 自动时间戳目录 | 工作目录。 |
| `SOC_VERSION` | `Ascend310B4` | 目标 SOC。 |
| `PRECISION_MODE` | `allow_fp32_to_fp16` | ATC 精度模式。 |
| `OM_BASENAME` | `full_where2_gather40slice_allowfp32tofp16` | OM 名称前缀。 |
| `COMPARE_DIR` | 自动时间戳目录 | 第三步建议输出目录。 |
| `LIMIT` | `10` | 第三步建议图片数。 |

### 2.7 `workflows/end2end_fullgraph_om_vis/run_all_minset_auto.sh`

用途：全自动流程（prepare + ATC + 等待 + 对比 + 汇总）。

可配置环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `IN_ONNX` | `$REPO_ROOT/ascend_c_onnx/end2end_simp.onnx` | 输入 ONNX。 |
| `WORK_DIR` | 自动时间戳目录 | 工作目录。 |
| `COMPARE_DIR` | 自动时间戳目录 | 对比输出目录。 |
| `SOC_VERSION` | `Ascend310B4` | 目标 SOC。 |
| `PRECISION_MODE` | `allow_fp32_to_fp16` | ATC 精度模式。 |
| `OM_BASENAME` | `full_minset_where2_gather40slice_allowfp32tofp16` | OM 名称前缀。 |
| `DEVICE_ID` | `0` | 设备 ID。 |
| `LIMIT` | `10` | 图片数量上限。 |
| `ATC_EXTRA_ARGS` | 空 | 附加 ATC 参数。 |
| `ORT_INTRA_OP_THREADS` | `8` | ORT intra-op 线程数。 |
| `ORT_INTER_OP_THREADS` | `1` | ORT inter-op 线程数。 |
| `BENCH_WARMUP_OM` | `1` | OM benchmark 预热次数。 |
| `BENCH_ITERS_OM` | `3` | OM benchmark 迭代次数。 |
| `BENCH_WARMUP_BOTH` | `0` | ORT+OM benchmark 预热次数。 |
| `BENCH_ITERS_BOTH` | `1` | ORT+OM benchmark 迭代次数。 |
| `POLL_SEC` | `15` | 轮询 OM 文件间隔秒数。 |
| `MAX_WAIT_SEC` | `0` | 最大等待秒数，`0` 表示不限时。 |
| `REUSE_OM_IF_EXISTS` | `1` | 若 OM 已存在是否复用。 |
| `SESSION_NAME` | 自动生成 | 覆盖默认 screen 会话名。 |

## 3. Python 脚本参数

### 3.1 `workflows/end2end_fullgraph_om_vis/prepare_fullgraph_onnx.py`

用途：生成 `atc_ready_full` 并可选执行 Where/Gather 替换。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--in-onnx` | `$REPO_ROOT/ascend_c_onnx/end2end_simp.onnx` | 输入 ONNX。 |
| `--out-dir` | 必填 | 输出目录。 |
| `--enable-where2-maskarith / --no-enable-where2-maskarith` | `True` | 是否替换 `Where_2`。 |
| `--where-name` | `/Where_2` | 目标 Where 节点名。 |
| `--pi2-name` | 空 | 手动指定 `mask*const` 常量名。 |
| `--enable-gather40-slice / --no-enable-gather40-slice` | `True` | 是否替换 `Gather_40`。 |
| `--gather-name` | `/Gather_40` | 目标 Gather 节点名。 |
| `--decompose-npu-layernorm / --no-decompose-npu-layernorm` | `True` | 是否分解 `NPULayerNorm`。 |

### 3.2 `scripts/patch_end2end_simp_pipeline.py`

用途：端到端 ONNX 改写总入口。

公共参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pipeline` | 必填 | 改写模式（见下）。 |
| `--in-onnx` | 必填 | 输入 ONNX。 |
| `--out-onnx` | 必填 | 输出 ONNX。 |
| `--opset` | `11` | TopK 替换等流程的目标 opset。 |
| `--nms-node-name` | `/NonMaxSuppression` | NMS 目标节点名。 |
| `--swap-boxes-xy` | `False` | NMS 替换时是否交换 box 的 xy。 |
| `--enable-npu-gelu` | `False` | 是否启用 NPUGelu 替换路径。 |
| `--decompose-npu-layernorm` | `False` | 在 `atc_ready_base/full` 中是否分解 NPULayerNorm。 |

`--pipeline` 可选值：
- `ops_to_npu`
- `decompose_npu_layernorm`
- `reduce_axes`
- `atc_ready_extras`
- `topk_to_ascend_topk`
- `nms_to_npu_nms_ort`
- `nms_squeeze_to_reshape`
- `atc_ready_base`
- `atc_ready_full`

### 3.3 `tools/patch_where2_maskarith.py`

用途：把一个 `Where` 改写为 `Cast + Mul + Sub`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--in-onnx` | 必填 | 输入 ONNX。 |
| `--out-onnx` | 必填 | 输出 ONNX。 |
| `--where-name` | `/Where_2` | 目标 Where 节点名。 |
| `--pi2-name` | 空 | 常量名；为空时自动从 else 分支 `Add(then,const)` 推断。 |

### 3.4 `tools/patch_gather40_to_slice_squeeze.py`

用途：把 `Gather(axis=1,index=-1)` 改写为 `Slice(4:5)+Squeeze`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--in-onnx` | 必填 | 输入 ONNX。 |
| `--out-onnx` | 必填 | 输出 ONNX。 |
| `--gather-name` | `/Gather_40` | 目标 Gather 节点名。 |

### 3.5 `scripts/run_om_only_with_cached_ort.py`

用途：仅跑 OM，尽量复用历史 ORT 缓存并生成可视化。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--om` | 必填 | OM 文件路径。 |
| `--image-dir` | `$REPO_ROOT/images` | 测试图片目录。 |
| `--input-hw` | `672,1024` | 预处理输入尺寸。 |
| `--limit` | `10` | 图片数量上限。 |
| `--device-id` | `0` | 设备 ID。 |
| `--cache-root` | `$REPO_ROOT/output` | ORT 缓存根目录。 |
| `--out-dir` | 必填 | 输出目录。 |
| `--single-image` | 空 | 单图 benchmark 图片名（为空取首图）。 |
| `--bench-warmup` | `10` | 单图 OM benchmark 预热次数。 |
| `--bench-iters` | `3` | 单图 OM benchmark 迭代次数。 |

### 3.6 `scripts/fill_ort_vis_and_benchmark.py`

用途：在 OM-only 结果基础上补跑 ORT、写对比报告和并排可视化。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--work-dir` | 必填 | `run_om_only_with_cached_ort.py` 输出目录。 |
| `--om` | 必填 | OM 路径（用于单图 ORT/OM benchmark）。 |
| `--image-dir` | `$REPO_ROOT/images` | 测试图片目录。 |
| `--input-hw` | `672,1024` | 预处理输入尺寸。 |
| `--onnx` | `$REPO_ROOT/ascend_c_onnx/end2end_simp.only_dets_labels.ort_tmp.onnx` | ORT ONNX。 |
| `--custom-ops` | `$REPO_ROOT/onnxruntime_mmrotate_ops/build/libonnxruntime_mmrotate_ops.so` | ORT 自定义算子库。 |
| `--ort-intra-op-threads` | `max(1,cpu_count/2)` | ORT intra-op 线程数。 |
| `--ort-inter-op-threads` | `1` | ORT inter-op 线程数。 |
| `--device-id` | `0` | 设备 ID。 |
| `--score-thr` | `0.3` | 可视化分数阈值。 |
| `--single-image` | 空 | 单图 benchmark 图片名（为空取首图）。 |
| `--bench-warmup` | `1` | 单图 ORT/OM benchmark 预热次数。 |
| `--bench-iters` | `3` | 单图 ORT/OM benchmark 迭代次数。 |

## 4. 推荐最小调用方式

```bash
REPO_ROOT=/path/to/open_source_release_20260222/standalone_end2end_simp_runner
cd "$REPO_ROOT"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base

SOC_VERSION=Ascend310B4 \
PRECISION_MODE=allow_fp32_to_fp16 \
LIMIT=10 \
bash workflows/end2end_fullgraph_om_vis/run_all_minset_auto.sh
```

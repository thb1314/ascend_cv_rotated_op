# End2End ONNX -> OM -> 运行对比可视化（统一流程）

这个目录把“从 `end2end_simp.onnx` 到 `OM` 再到 `OM/ORT` 对比可视化”的相关脚本集中管理，避免在多个目录分散执行。

脚本参数速查见：
- `standalone_end2end_simp_runner/README_SCRIPT_ARGS.md`

## 1. 目录结构

- `prepare_fullgraph_onnx.py`
  - 作用：从 `end2end_simp.onnx` 生成 `atc_ready_full`，并按需要做算子替换（`Where_2`、`Gather_40`）。
  - 默认会启用 `--decompose-npu-layernorm`，减少对 `NPULayerNorm` 自定义算子的依赖。
- `build_full_om_in_screen.sh`
  - 作用：生成 `atc` 命令脚本，并通过 `tools/run_atc_in_screen.sh` 在 `screen` 后台编译 `OM`。
- `run_compare_and_visualize.sh`
  - 作用：跑 `OM` 推理、补跑 `ORT`、输出测速、误差对比和每张图 `vis_side_by_side.jpg`。
- `run_all.sh`
  - 作用：一键串起前两步（生成 ONNX + screen 启动 ATC），并提示第三步命令。

---

## 2. 原理说明（重点）

### 2.1 为什么先做 `atc_ready_full`

`scripts/patch_end2end_simp_pipeline.py --pipeline atc_ready_full` 会把原始 `end2end_simp.onnx` 做成更适合 Ascend ATC 编译和运行的图（包含 NPU 侧适配）。

### 2.2 `/Where_2` 替换（`Where -> Cast+Mul+Sub`）

脚本：`tools/patch_where2_maskarith.py`  
目的：避免 `Where` 在该子图上的实现/精度行为不稳定，改成等价算术形式，减少后端不确定性。

### 2.3 `Gather` 替换（`Gather(axis=1,-1,last_dim=5) -> Slice+Squeeze`）

脚本：`tools/patch_gather40_to_slice_squeeze.py`  
默认自动匹配条件：`axis=1`、索引为标量 `-1`、输入最后一维 `=5`；也支持显式 `--gather-name`。  
目的：规避该模式 `Gather` 在 NPU 上触发 `te_gatherv2` 运行异常风险（历史上出现过 `507011`）。

### 2.4 为什么 `ATC` 用 `screen`

整图编译耗时长、机器资源紧张时容易中断。`screen` 能保证 SSH 断开后编译继续执行，并保留日志。

### 2.5 对比与可视化策略

1. 先跑 `OM`（可复用历史 ORT 缓存，快速得到 OM 结果）。
2. 再补跑真实 `ORT`，统一输出：
   - 速度对比（`summary_ort_vs_om.csv`）
   - 误差指标（`dets_max_abs`, `dets_mean_abs`, `labels_neq`）
   - 可视化（每张图 `vis_ort.jpg`, `vis_om.jpg`, `vis_side_by_side.jpg`）

---

## 3. 详细操作步骤 + Shell 命令

以下命令在仓库根目录执行（`REPO_ROOT` 指向 `standalone_end2end_simp_runner` 目录）：

### Step A：准备 ONNX（含替换）

```bash
REPO_ROOT=/path/to/open_source_release_20260222/standalone_end2end_simp_runner
cd "$REPO_ROOT"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source env.sh

WORK_DIR="$REPO_ROOT/output/fullgraph_pipeline_demo"
python workflows/end2end_fullgraph_om_vis/prepare_fullgraph_onnx.py \
  --in-onnx "$REPO_ROOT/ascend_c_onnx/end2end_simp.onnx" \
  --out-dir "$WORK_DIR" \
  --enable-where2-maskarith \
  --enable-gather40-slice
```

输出：
- `WORK_DIR/fullgraph_prepare_manifest.json`
- `WORK_DIR/npu_full.atc_ready_full*.onnx`

### Step B：screen + ATC 编译 OM

```bash
REPO_ROOT=/path/to/open_source_release_20260222/standalone_end2end_simp_runner
cd "$REPO_ROOT"
WORK_DIR="$REPO_ROOT/output/fullgraph_pipeline_demo"
MODEL_ONNX=$WORK_DIR/npu_full.atc_ready_full.where2_maskarith.gather40_slice.onnx

WORK_DIR="$WORK_DIR" \
MODEL_ONNX="$MODEL_ONNX" \
SOC_VERSION=Ascend310B4 \
PRECISION_MODE=allow_fp32_to_fp16 \
OM_BASENAME=full_where2_gather40slice_allowfp32tofp16 \
bash workflows/end2end_fullgraph_om_vis/build_full_om_in_screen.sh
```

监控：

```bash
screen -ls
tail -f $WORK_DIR/screen_atc_full_where2_gather40slice_allowfp32tofp16.log
```

编译产物：
- `WORK_DIR/full_where2_gather40slice_allowfp32tofp16.om`

### Step C：运行 OM/ORT 对比 + 可视化 + 测速

```bash
REPO_ROOT=/path/to/open_source_release_20260222/standalone_end2end_simp_runner
cd "$REPO_ROOT"
OM_PATH="$REPO_ROOT/output/fullgraph_pipeline_demo/full_where2_gather40slice_allowfp32tofp16.om"
OUT_DIR="$REPO_ROOT/output/fullgraph_compare_demo"

OM_PATH="$OM_PATH" \
OUT_DIR="$OUT_DIR" \
LIMIT=10 \
ORT_INTRA_OP_THREADS=8 \
bash workflows/end2end_fullgraph_om_vis/run_compare_and_visualize.sh
```

输出：
- `OUT_DIR/summary_om_only.csv`
- `OUT_DIR/summary_ort_vs_om.csv`
- `OUT_DIR/report_ort_vs_om.json`
- `OUT_DIR/<image_stem>/vis_side_by_side.jpg`（每张图都有）

---

## 4. 一键入口（推荐）

```bash
REPO_ROOT=/path/to/open_source_release_20260222/standalone_end2end_simp_runner
cd "$REPO_ROOT"
IN_ONNX="$REPO_ROOT/ascend_c_onnx/end2end_simp.onnx" \
WORK_DIR="$REPO_ROOT/output/fullgraph_pipeline_demo" \
SOC_VERSION=Ascend310B4 \
PRECISION_MODE=allow_fp32_to_fp16 \
bash workflows/end2end_fullgraph_om_vis/run_all.sh
```

说明：`run_all.sh` 会执行 Step A + Step B，并提示你在 ATC 完成后执行 Step C 命令。

### 全自动一键（最小算子集）

如果希望一条命令自动完成：
1) 准备 ONNX（含 `decompose_npu_layernorm` + where/gather 替换）  
2) `screen+ATC` 编译并等待 `OM` 生成  
3) `OM/ORT` 对比、测速、可视化  

可直接执行：

```bash
REPO_ROOT=/path/to/open_source_release_20260222/standalone_end2end_simp_runner
cd "$REPO_ROOT"
IN_ONNX="$REPO_ROOT/ascend_c_onnx/end2end_simp.onnx" \
WORK_DIR="$REPO_ROOT/output/fullgraph_pipeline_minset_auto_demo" \
COMPARE_DIR="$REPO_ROOT/output/fullgraph_compare_minset_auto_demo" \
SOC_VERSION=Ascend310B4 \
PRECISION_MODE=allow_fp32_to_fp16 \
LIMIT=10 \
bash workflows/end2end_fullgraph_om_vis/run_all_minset_auto.sh
```

关键产物：
- `WORK_DIR/used_custom_ops.txt`：当前整图实际用到的自定义算子清单
- `COMPARE_DIR/summary_ort_vs_om.csv`
- `COMPARE_DIR/report_ort_vs_om.json`
- `COMPARE_DIR/<image_stem>/vis_side_by_side.jpg`

---

## 5. 常见参数

- `PRECISION_MODE`
  - 常用：`allow_fp32_to_fp16`
  - 可切：`force_fp32`
- `ATC_EXTRA_ARGS`（可选）
  - 例如：`--deterministic=1 --op_select_implmode=high_precision_for_all`
- `LIMIT`
  - 控制图片数量（ORT 很慢时建议先小批量）。

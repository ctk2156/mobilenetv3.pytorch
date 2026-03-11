# TFLite Benchmark 编译与对比说明

## 目标

这里我们只对比：

- 开启 XNNPACK delegate
- 单线程
- `fp32`
- `fp16(force)`

对比对象：

1. baseline:
   `artifacts/models/tflite/shufflenetv2_1.0x/shufflenetv2_1.0x_float32.tflite`
2. optimized:
   `artifacts/models/tflite/shufflenetv2_1.0x_custom_cs/shufflenetv2_1.0x_custom_cs_float32.tflite`

## 1. benchmark 二进制

本项目使用的 benchmark 目标：

```bash
bazel build //tensorflow/lite/tools/benchmark:benchmark_model_cpu_xnnpack_custom_ops
```

原因：

- 只走 CPU + XNNPACK 路径
- 同时注册了 `ChannelShuffleSplit` 自定义算子

如果只用普通 `benchmark_model`，在有 custom op 的情况下不够方便。

## 2. 编译命令

在 `tensorflow/` 仓库根目录下执行：

```bash
bazel build //tensorflow/lite/tools/benchmark:benchmark_model_cpu_xnnpack_custom_ops
```

编译完成后的二进制：

```bash
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_cpu_xnnpack_custom_ops
```

## 3. benchmark 命令

### baseline + XNNPACK fp32

```bash
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_cpu_xnnpack_custom_ops \
  --graph=/Users/zhangdongjiu/Workspaces/mobilenetv3.pytorch/artifacts/models/tflite/shufflenetv2_1.0x/shufflenetv2_1.0x_float32.tflite \
  --num_threads=1 \
  --num_runs=20 \
  --warmup_runs=5 \
  --use_xnnpack=true
```

### baseline + XNNPACK fp16

```bash
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_cpu_xnnpack_custom_ops \
  --graph=/Users/zhangdongjiu/Workspaces/mobilenetv3.pytorch/artifacts/models/tflite/shufflenetv2_1.0x/shufflenetv2_1.0x_float32.tflite \
  --num_threads=1 \
  --num_runs=20 \
  --warmup_runs=5 \
  --use_xnnpack=true \
  --xnnpack_force_fp16=true
```

### optimized + XNNPACK fp32

```bash
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_cpu_xnnpack_custom_ops \
  --graph=/Users/zhangdongjiu/Workspaces/mobilenetv3.pytorch/artifacts/models/tflite/shufflenetv2_1.0x_custom_cs/shufflenetv2_1.0x_custom_cs_float32.tflite \
  --num_threads=1 \
  --num_runs=20 \
  --warmup_runs=5 \
  --use_xnnpack=true
```

### optimized + XNNPACK fp16

```bash
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_cpu_xnnpack_custom_ops \
  --graph=/Users/zhangdongjiu/Workspaces/mobilenetv3.pytorch/artifacts/models/tflite/shufflenetv2_1.0x_custom_cs/shufflenetv2_1.0x_custom_cs_float32.tflite \
  --num_threads=1 \
  --num_runs=20 \
  --warmup_runs=5 \
  --use_xnnpack=true \
  --xnnpack_force_fp16=true
```

## 4. 当前实测结果

测试环境：

- Apple Silicon
- 单线程
- `use_xnnpack=true`
- `num_runs=20`
- `warmup_runs=5`

结果：

| 模型 | 精度策略 | 平均耗时(us) |
| --- | --- | ---: |
| baseline | fp32 | 5337.25 |
| optimized | fp32 | 5039.34 |
| baseline | fp16(force) | 2792.91 |
| optimized | fp16(force) | 2497.45 |

### 相对 baseline 的收益

fp32:

- 绝对减少：`297.91 us`
- 相对提升：`5.58%`

fp16:

- 绝对减少：`295.46 us`
- 相对提升：`10.58%`

### optimized 模型自身的 fp16 收益

同一份 optimized 模型：

- fp32: `5039.34 us`
- fp16(force): `2497.45 us`

也就是：

- 绝对减少：`2541.89 us`
- 相对提升：约 `50.44%`

## 5. 如何判断 benchmark 是否真的走了 XNNPACK

看日志里是否出现：

```text
Created TensorFlow Lite XNNPACK delegate for CPU.
Explicitly applied XNNPACK delegate ...
```

如果是完整委托，还会看到：

```text
the model graph will be completely executed by the delegate
```

## 6. 如何做 baseline / optimized 对比

建议固定以下参数不变：

- `--num_threads=1`
- `--num_runs=20`
- `--warmup_runs=5`
- `--use_xnnpack=true`

只切换：

- `--graph=...baseline...`
- `--graph=...optimized...`
- 是否增加 `--xnnpack_force_fp16=true`

这样口径最稳定。

## 7. 当前 profiling 结论

对 optimized 模型做 `--enable_op_profiling=true` 后，当前最大瓶颈不是 channel shuffle，而是 pointwise 1x1 GEMM。

大致占比：

- `Fully Connected (NC, F32) GEMM`: 约 86%
- `Depthwise Conv`: 约 5.5%
- `Channel Shuffle (NHWC, X32)`: 约 1.2%

所以后续如果继续冲性能，最值得做的是：

1. `channel shuffle + 后继 1x1 conv` 融合
2. 继续优化 pointwise GEMM 路径

# ShuffleNetV2 从 PyTorch 到 ONNX 再到 TFLite 的导出流程

## 总览

当前导出流程分三步：

1. PyTorch baseline / custom 模型构建
2. 导出 ONNX
3. 通过轻量 flatbuffer_direct 流程生成 TFLite

主入口脚本：

- `scripts/export_shufflenetv2_custom_tflite.py`

## 1. PyTorch 模型阶段

相关文件：

- `shufflenetv2.py`
- `shufflenetv2_custom.py`

### baseline 模型

- 使用 Megvii 风格 ShuffleNetV2 实现
- 普通 channel shuffle 逻辑仍然在 `NCHW`

### custom 模型

- stride=1 block 不再调用原始的 `channel_shuffle(x)`
- 改为显式双输入双输出的 `ChannelShuffleSplit`

也就是把原来的：

```text
x -> channel_shuffle -> (x_proj, x_main)
```

重写成：

```text
(left, right) -> ChannelShuffleSplit(left, right) -> (x_proj, x_main)
```

这样更适合在 ONNX/TFLite/XNNPACK 里保留成一个独立算子。

## 2. baseline 权重迁移

在 `scripts/export_shufflenetv2_custom_tflite.py` 中：

- 先构造 baseline 模型
- 再构造 custom 模型
- 用 `transfer_state_dict_by_order(...)` 按参数顺序把权重搬过去

这样做的前提是：

- custom 模型只是改了 channel shuffle 表达方式
- 卷积 / BN / FC 的参数排列没有被改变

所以我们可以在不重新训练的情况下直接得到数值等价模型。

脚本里也做了一个简单验证：

- `validate_equivalence(...)`
- 当前结果是 `Max abs diff vs baseline ShuffleNetV2: 0.0`

## 3. PyTorch -> ONNX

脚本使用：

```python
torch.onnx.export(...)
```

关键点在 `ChannelShuffleSplitFunction.symbolic(...)`：

1. 先把两个 `NCHW` 输入转成 `NHWC`
2. 输出自定义 `ChannelShuffleSplit` ONNX 节点
3. 再把两个输出转回 `NCHW`

所以 ONNX 中实际的 custom node 是 `NHWC` 语义。

### 导出的 custom op

PyTorch/ONNX 侧：

- `shufflenet_codex::ChannelShuffleSplit`

TFLite custom code：

- `ONNX_CHANNELSHUFFLESPLIT`

## 4. ONNX shape 修复

ONNX 对自定义算子的 shape 传播不稳定，尤其在后半段 block 上容易丢失 shape。

为了解决这个问题，导出脚本里增加了：

- `repair_channel_shuffle_split_shapes(...)`

它会在 ONNX 导出后做两类修复：

1. 修复 `Transpose` 节点的输出 shape
2. 修复 `ChannelShuffleSplit` 节点的输出 shape

这是让后续 flatbuffer_direct 降级时保住静态 shape 的第一步。

## 5. ONNX -> TFLite

相关文件：

- `scripts/tflite_export_utils.py`
- `onnx2tf/onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/onnx2tf/tflite_builder/schema_loader.py`

### 为什么没有直接走 `from onnx2tf import convert`

本机环境里，`tensorflow` 与 `onnx` 的某些顶层 import 组合会触发原生库冲突。

所以这里没有走重量级入口，而是改成了轻量 pipeline：

1. fake package 注入，避免触发 `onnx2tf` 顶层重量级 `__init__`
2. 只 import `tflite_builder` 所需模块
3. 跑 preprocess
4. 跑 `lower_onnx_to_ir`
5. 修 IR
6. 写 flatbuffer

### 轻量导出流程

`convert_onnx_to_fp32_tflite(...)` 现在做的是：

1. `onnx.load(...)`
2. `run_preprocess_pipeline(...)`
3. `lower_onnx_to_ir(...)`
4. `_repair_model_ir_shapes(...)`
5. `_optimize_padv2_pool_patterns(...)`
6. 再次 `_repair_model_ir_shapes(...)`
7. `clone_model_ir_with_float32(...)`
8. `prune_identity_cast_operators(...)`
9. `optimize_redundant_transpose_operators(...)`
10. `write_model_file(...)`
11. `build_tensor_correspondence_report(...)`

## 6. 自定义算子如何保留下来

在导出参数里：

- `flatbuffer_direct_allow_custom_ops=True`
- `flatbuffer_direct_custom_op_allowlist=["ChannelShuffleSplit"]`

在 `onnx2tf` 的 `flatbuffer_direct` lowering 中：

- `ChannelShuffleSplit` 不会被降成普通 builtin
- 会作为一个 CUSTOM op 写入 flatbuffer
- custom code 为 `ONNX_CHANNELSHUFFLESPLIT`

## 7. TFLite custom kernel

文件：

- `tensorflow/tensorflow/lite/kernels/channel_shuffle_split_custom.cc`

作用：

- 作为 non-delegate fallback 路径
- 在 `--use_xnnpack=false` 时也能正确执行

当前实现是 `NHWC` 语义。

## 8. TFLite XNNPACK delegate

文件：

- `tensorflow/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc`

当前逻辑：

- 识别 `ONNX_CHANNELSHUFFLESPLIT`
- 直接调用 `xnn_define_channel_shuffle(...)`

所以导出后的优化模型在 delegate 路径下不会再把 channel shuffle 展开成一串基础算子。

## 9. 最终正式产物

正式优化模型：

- ONNX:
  `artifacts/models/onnx/shufflenetv2_1.0x_custom_cs.onnx`
- TFLite:
  `artifacts/models/tflite/shufflenetv2_1.0x_custom_cs/shufflenetv2_1.0x_custom_cs_float32.tflite`

baseline 模型：

- ONNX:
  `artifacts/models/onnx/shufflenetv2_1.0x.onnx`
- TFLite:
  `artifacts/models/tflite/shufflenetv2_1.0x/shufflenetv2_1.0x_float32.tflite`

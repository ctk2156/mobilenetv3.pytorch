# ShuffleNetV2 自定义 XNNPACK 实现细节

## 目标

这个版本的优化目标有两件事：

1. 把 ShuffleNetV2 里的 channel shuffle 从 `NCHW` 语义切到 `NHWC` 语义，彻底去掉围绕 custom op 的 layout bridge。
2. 把原来由多个基础算子拼出来的数据搬运子图，收敛成一个原生的 XNNPACK `Channel Shuffle (NHWC)` 算子。

最终模型：

- baseline: `artifacts/models/tflite/shufflenetv2_1.0x/shufflenetv2_1.0x_float32.tflite`
- optimized: `artifacts/models/tflite/shufflenetv2_1.0x_custom_cs/shufflenetv2_1.0x_custom_cs_float32.tflite`

## 整体架构

整条链路现在分成四层：

1. PyTorch 模型层
2. ONNX custom op 导出层
3. TFLite custom kernel 层
4. TFLite XNNPACK delegate / XNNPACK native operator 层

### 1. PyTorch 模型层

文件：

- `shufflenetv2_custom.py`

做法：

- 仍然保留 PyTorch 前向里的 `NCHW` 语义，保证数值逻辑和原始 ShuffleNetV2 一致。
- 在 `symbolic()` 里显式插入：
  - `NCHW -> NHWC` 的 `Transpose`
  - `ChannelShuffleSplit` custom op
  - `NHWC -> NCHW` 的 `Transpose`

这样做的原因是：

- 训练/推理时 PyTorch 侧仍然是熟悉的 `NCHW`
- 导出后的 ONNX custom op 本体已经是 `NHWC`
- 后续 `onnx -> tflite` 转换时不再需要为了 custom op 额外插一圈 layout bridge

### 2. ONNX custom op 导出层

导出的 custom op 名称：

- `shufflenet_codex::ChannelShuffleSplit`

在 TFLite 里最终映射为：

- `ONNX_CHANNELSHUFFLESPLIT`

当前 custom op 的语义是：

- 输入：两个 `NHWC` 张量 `A` 和 `B`
- 约束：
  - `N/H/W` 相同
  - `C_a + C_b` 为偶数
- 行为：
  - 按 channel 维拼接 `A` 和 `B`
  - 偶数 channel 写到 output0
  - 奇数 channel 写到 output1

也就是：

```text
concat(A, B, axis=C) -> merged
output0 = merged[..., 0::2]
output1 = merged[..., 1::2]
```

### 3. TFLite custom kernel 层

文件：

- `tensorflow/tensorflow/lite/kernels/channel_shuffle_split_custom.cc`

这个 kernel 现在已经改成 `NHWC` 版本：

- 校验维度按 `[N, H, W, C]`
- 输出 shape 也是 `[N, H, W, (C_a + C_b) / 2]`
- 内层实现按 pixel 遍历，然后在 channel 维做 even/odd 拆分

它的作用主要是：

- 保证在不走 XNNPACK delegate 的时候，模型仍然能正确执行
- 给 benchmark 中 `--use_xnnpack=false` 的对照测试提供可用路径

### 4. XNNPACK delegate / native operator 层

相关文件：

- `tensorflow/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc`
- `tensorflow/third_party/XNNPACK/src/subgraph/channel-shuffle.c`
- `tensorflow/third_party/XNNPACK/src/operators/channel-shuffle-nhwc.c`
- `tensorflow/third_party/XNNPACK/src/operator-run.c`
- `tensorflow/third_party/XNNPACK/src/runtime.c`

这里是本次优化的核心。

#### 4.1 Delegate 侧

早期版本里，delegate 会把 `ChannelShuffleSplit` 展开成一串基础算子。

现在改成：

- `VisitChannelShuffleSplitNode(...)`
- 直接调用 `xnn_define_channel_shuffle(...)`

所以 TFLite delegate 层不再把它拆成：

- `concat`
- `reshape`
- `transpose`
- `even_split`

而是直接构造一个原生 XNNPACK node。

#### 4.2 XNNPACK subgraph 层

在 `src/subgraph/channel-shuffle.c` 中新增了：

- `xnn_define_channel_shuffle(...)`
- `create_channel_shuffle_operator(...)`
- `reshape_channel_shuffle_operator(...)`
- `setup_channel_shuffle_operator(...)`

这意味着 channel shuffle 已经进入 XNNPACK 的标准生命周期：

1. define
2. create
3. reshape
4. setup
5. run

#### 4.3 XNNPACK operator 层

当前已经有两套 operator：

- `xnn_operator_type_channel_shuffle_nhwc_x32`
- `xnn_operator_type_channel_shuffle_nhwc_x16`

对应接口：

- `xnn_create_channel_shuffle_nhwc_x32`
- `xnn_reshape_channel_shuffle_nhwc_x32`
- `xnn_setup_channel_shuffle_nhwc_x32`
- `xnn_create_channel_shuffle_nhwc_x16`
- `xnn_reshape_channel_shuffle_nhwc_x16`
- `xnn_setup_channel_shuffle_nhwc_x16`

它们分别服务于：

- `fp32` 路径
- `fp16 force` 路径

#### 4.4 Kernel 层

在 `src/operators/channel-shuffle-nhwc.c` 里：

- scalar fallback 已实现
- ARM NEON 快路径已实现

目前保留的有效优化是：

- `x32` 路径做了 16-channel 展开
- task tile 做了按 channel 数自适应的调度大小选择

另外，为了让 `fp16` 也真正可用，又补了：

- `x16` scalar path
- `x16` NEON path

## 目前已经消掉的额外开销

### 1. 围绕 custom op 的 transpose

这是第一阶段最大的收益来源。

在旧版本里，`ChannelShuffleSplit` 是 `NCHW` 语义，导致：

- Conv 输出先转成 `NCHW`
- custom op 做完再转回 `NHWC`

现在这圈桥已经去掉了。

### 2. 用一个原生 XNNPACK node 替代一串搬运子图

现在 channel shuffle 在 delegate 下不再是“多个基础节点拼出来的逻辑”，而是一个独立的原生算子。

## 当前性能结论

从整图 profiling 看：

- pointwise 1x1 GEMM 仍然是主瓶颈
- channel shuffle 现在只占整图执行时间的大约 1% 出头

这意味着：

- channel shuffle 本身已经不再是主要瓶颈
- 继续只抠 channel shuffle kernel，收益空间很有限

## 还有哪些进一步优化空间

### 1. 把 channel shuffle 融到后继 1x1 conv

这是最值得继续做的一步。

因为当前整图里：

- 1x1 pointwise GEMM 占比接近 86%
- channel shuffle 只占大约 1.2%

如果让后继 pointwise conv 直接吃“逻辑上的 shuffled channel 顺序”，会比继续优化独立 shuffle 算子更值。

### 2. 针对固定通道数做专门 fast path

ShuffleNetV2 1.0x 里主路径反复出现：

- 58
- 116
- 232

可以继续为这些固定 channel count 做更专门的 NEON fast path。

### 3. 进一步利用 fp16

目前 `fp16 force` 路径已经跑通，收益明显，后续可以继续围绕：

- `fp16` 下的 pointwise GEMM
- `fp16` 下的内存带宽

做更大规模优化。

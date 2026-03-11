"""ShuffleNetV2 with a custom dual-input dual-output ChannelShuffleSplit op."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "ChannelShuffleSplit",
    "ShuffleNetV2CustomChannelShuffle",
    "shufflenetv2_custom_channel_shuffle",
]


def _set_output_type_nchw(value: torch.Value, reference: torch.Value, channels: int) -> None:
    tensor_type = reference.type()
    sizes = getattr(tensor_type, "sizes", lambda: None)()
    if sizes is None or len(sizes) != 4:
        return
    batch, _, height, width = sizes
    value.setType(tensor_type.with_sizes([batch, channels, height, width]))


def _set_output_type_nhwc(value: torch.Value, reference: torch.Value, channels: int) -> None:
    tensor_type = reference.type()
    sizes = getattr(tensor_type, "sizes", lambda: None)()
    if sizes is None or len(sizes) != 4:
        return
    batch, _, height, width = sizes
    value.setType(tensor_type.with_sizes([batch, height, width, channels]))


class ChannelShuffleSplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_a: torch.Tensor, input_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_a.ndim != 4 or input_b.ndim != 4:
            raise ValueError("ChannelShuffleSplit expects two 4D NCHW tensors.")
        if input_a.shape[0] != input_b.shape[0] or input_a.shape[2:] != input_b.shape[2:]:
            raise ValueError("ChannelShuffleSplit inputs must match in batch and spatial dims.")

        merged = torch.cat((input_a, input_b), dim=1)
        batch_size, channels, height, width = merged.shape
        if channels % 2 != 0:
            raise ValueError(f"Merged channel count must be even, got {channels}.")
        shuffled = merged.reshape(batch_size, channels // 2, 2, height, width)
        shuffled = shuffled.permute(0, 2, 1, 3, 4).contiguous()
        return shuffled[:, 0], shuffled[:, 1]

    @staticmethod
    def symbolic(g, input_a: torch.Value, input_b: torch.Value):
        input_a_nhwc = g.op("Transpose", input_a, perm_i=[0, 2, 3, 1])
        input_b_nhwc = g.op("Transpose", input_b, perm_i=[0, 2, 3, 1])
        output_a_nhwc, output_b_nhwc = g.op(
            "shufflenet_codex::ChannelShuffleSplit",
            input_a_nhwc,
            input_b_nhwc,
            outputs=2,
        )
        sizes_a = getattr(input_a.type(), "sizes", lambda: None)()
        sizes_b = getattr(input_b.type(), "sizes", lambda: None)()
        if (
            sizes_a is not None
            and sizes_b is not None
            and len(sizes_a) == 4
            and len(sizes_b) == 4
            and sizes_a[1] is not None
            and sizes_b[1] is not None
        ):
            out_channels = int(sizes_a[1]) + int(sizes_b[1])
            if out_channels % 2 == 0:
                _set_output_type_nhwc(input_a_nhwc, input_a, int(sizes_a[1]))
                _set_output_type_nhwc(input_b_nhwc, input_b, int(sizes_b[1]))
                _set_output_type_nhwc(output_a_nhwc, input_a, out_channels // 2)
                _set_output_type_nhwc(output_b_nhwc, input_a, out_channels // 2)
        output_a = g.op("Transpose", output_a_nhwc, perm_i=[0, 3, 1, 2])
        output_b = g.op("Transpose", output_b_nhwc, perm_i=[0, 3, 1, 2])
        if (
            sizes_a is not None
            and sizes_b is not None
            and len(sizes_a) == 4
            and len(sizes_b) == 4
            and sizes_a[1] is not None
            and sizes_b[1] is not None
        ):
            out_channels = int(sizes_a[1]) + int(sizes_b[1])
            if out_channels % 2 == 0:
                _set_output_type_nchw(output_a, input_a, out_channels // 2)
                _set_output_type_nchw(output_b, input_a, out_channels // 2)
        return output_a, output_b


class ChannelShuffleSplit(nn.Module):
    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return ChannelShuffleSplitFunction.apply(input_a, input_b)


class ShuffleV2BlockStride2(nn.Module):
    def __init__(self, inp: int, oup: int, mid_channels: int, *, ksize: int) -> None:
        super().__init__()
        pad = ksize // 2
        outputs = oup - inp
        self.branch_main = nn.Sequential(
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                ksize,
                2,
                pad,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )
        self.branch_proj = nn.Sequential(
            nn.Conv2d(inp, inp, ksize, 2, pad, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.branch_proj(x), self.branch_main(x)


class ShuffleV2BlockStride1(nn.Module):
    def __init__(self, inp_half: int, oup: int, mid_channels: int, *, ksize: int) -> None:
        super().__init__()
        pad = ksize // 2
        outputs = oup - inp_half
        if outputs != inp_half:
            raise ValueError(
                f"Stride-1 ShuffleNetV2 block expects equal halves, got inp_half={inp_half} oup={oup}."
            )
        self.channel_shuffle = ChannelShuffleSplit()
        self.branch_main = nn.Sequential(
            nn.Conv2d(inp_half, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                ksize,
                1,
                pad,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, input_a: torch.Tensor, input_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_proj, x = self.channel_shuffle(input_a, input_b)
        return x_proj, self.branch_main(x)


class ShuffleV2Stage(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, repeats: int) -> None:
        super().__init__()
        if repeats < 1:
            raise ValueError("ShuffleV2Stage requires at least one block.")

        self.first_block = ShuffleV2BlockStride2(
            input_channel,
            output_channel,
            mid_channels=output_channel // 2,
            ksize=3,
        )
        blocks = []
        half_channel = output_channel // 2
        for _ in range(repeats - 1):
            blocks.append(
                ShuffleV2BlockStride1(
                    half_channel,
                    output_channel,
                    mid_channels=output_channel // 2,
                    ksize=3,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.first_block(x)
        for block in self.blocks:
            left, right = block(left, right)
        return torch.cat((left, right), dim=1)


class ShuffleNetV2CustomChannelShuffle(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 224,
        num_classes: int = 1000,
        model_size: str = "1.0x",
    ) -> None:
        super().__init__()
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size

        if model_size == "0.5x":
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError(f"Unsupported model_size: {model_size}")

        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stages = []
        for stage_index, repeats in enumerate(self.stage_repeats):
            output_channel = self.stage_out_channels[stage_index + 2]
            stages.append(ShuffleV2Stage(input_channel, output_channel, repeats))
            input_channel = output_channel
        self.stages = nn.ModuleList(stages)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(max(1, input_size // 32))
        if self.model_size == "2.0x":
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.stage_out_channels[-1], num_classes, bias=False)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.maxpool(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        if self.model_size == "2.0x":
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(module.weight, 0, 0.01)
                else:
                    nn.init.normal_(module.weight, 0, 1.0 / module.weight.shape[1])
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def shufflenetv2_custom_channel_shuffle(**kwargs: object) -> ShuffleNetV2CustomChannelShuffle:
    return ShuffleNetV2CustomChannelShuffle(**kwargs)

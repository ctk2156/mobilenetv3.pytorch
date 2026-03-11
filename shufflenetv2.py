"""ShuffleNetV2 based on the official Megvii reference implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ShuffleNetV2", "shufflenetv2"]


class ShuffleV2Block(nn.Module):
    def __init__(self, inp: int, oup: int, mid_channels: int, *, ksize: int, stride: int) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"Unsupported stride: {stride}")

        pad = ksize // 2
        outputs = oup - inp

        self.stride = int(stride)
        self.branch_main = nn.Sequential(
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                ksize,
                stride,
                pad,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

        if self.stride == 2:
            self.branch_proj = nn.Sequential(
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x_proj, x_main = self.channel_shuffle(x)
            return torch.cat((x_proj, self.branch_main(x_main)), dim=1)
        return torch.cat((self.branch_proj(x), self.branch_main(x)), dim=1)

    @staticmethod
    def channel_shuffle(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = x.size()
        if num_channels % 4 != 0:
            raise ValueError(f"Channel count must be divisible by 4, got {num_channels}")
        x = x.reshape(batch_size * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
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

        features: list[nn.Module] = []
        for stage_index, num_repeat in enumerate(self.stage_repeats):
            output_channel = self.stage_out_channels[stage_index + 2]
            for repeat_index in range(num_repeat):
                if repeat_index == 0:
                    features.append(
                        ShuffleV2Block(
                            input_channel,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=2,
                        )
                    )
                else:
                    features.append(
                        ShuffleV2Block(
                            input_channel // 2,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=1,
                        )
                    )
                input_channel = output_channel
        self.features = nn.Sequential(*features)

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
        x = self.features(x)
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


def shufflenetv2(**kwargs: object) -> ShuffleNetV2:
    return ShuffleNetV2(**kwargs)

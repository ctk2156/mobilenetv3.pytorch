#!/usr/bin/env python3
"""Export ShuffleNetV2 to ONNX and fp32 TFLite artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shufflenetv2 import shufflenetv2
from tflite_export_utils import convert_onnx_to_fp32_tflite, format_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-size",
        choices=["0.5x", "1.0x", "1.5x", "2.0x"],
        default="1.0x",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "models",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_output_name(args: argparse.Namespace) -> str:
    if args.output_name:
        return args.output_name
    return f"shufflenetv2_{args.model_size}"


def main() -> int:
    args = parse_args()
    output_name = resolve_output_name(args)
    artifacts_root = args.artifacts_root.resolve()
    onnx_dir = artifacts_root / "onnx"
    onnx_path = onnx_dir / f"{output_name}.onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        if onnx_path.exists():
            onnx_path.unlink()
    elif onnx_path.exists():
        raise SystemExit(f"Refusing to overwrite existing ONNX file: {onnx_path}")

    model = shufflenetv2(
        input_size=int(args.image_size),
        num_classes=int(args.num_classes),
        model_size=args.model_size,
    ).eval()
    example = torch.randn(1, 3, int(args.image_size), int(args.image_size), dtype=torch.float32)

    print(f"[1/2] Exporting ONNX -> {onnx_path}")
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"[2/2] Converting ONNX -> TFLite ({output_name})")
    payload = convert_onnx_to_fp32_tflite(
        onnx_path=onnx_path,
        output_name=output_name,
        input_height=int(args.image_size),
        input_width=int(args.image_size),
        artifacts_root=artifacts_root,
        force=bool(args.force),
        enable_accumulation_type_float16=False,
        output_signaturedefs=False,
        metadata={
            "model_impl": "official_megvii_shufflenetv2",
            "model_size": args.model_size,
            "weights": "random",
        },
    )

    print("Export complete:")
    print(f"  ONNX    {onnx_path} ({format_size(onnx_path)})")
    print(f"  FP32    {payload['float32_tflite']} ({format_size(Path(payload['float32_tflite']))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

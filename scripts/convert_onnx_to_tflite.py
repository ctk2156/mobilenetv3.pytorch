#!/usr/bin/env python3
"""Convert an ONNX model to fp32 TFLite and drop generated fp16 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tflite_export_utils import convert_onnx_to_fp32_tflite, format_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/models"),
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="{}",
        help="Additional export metadata as a JSON object.",
    )
    parser.add_argument(
        "--enable-accumulation-type-float16",
        action="store_true",
    )
    parser.add_argument(
        "--output-signaturedefs",
        action="store_true",
    )
    parser.add_argument(
        "--flatbuffer-direct-allow-custom-ops",
        action="store_true",
    )
    parser.add_argument(
        "--flatbuffer-direct-custom-op-allowlist",
        type=str,
        nargs="*",
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata_json)
    payload = convert_onnx_to_fp32_tflite(
        onnx_path=args.onnx_path,
        output_name=args.output_name,
        input_height=int(args.input_height),
        input_width=int(args.input_width),
        artifacts_root=args.artifacts_root,
        force=bool(args.force),
        enable_accumulation_type_float16=bool(args.enable_accumulation_type_float16),
        output_signaturedefs=bool(args.output_signaturedefs),
        flatbuffer_direct_allow_custom_ops=bool(
            args.flatbuffer_direct_allow_custom_ops
        ),
        flatbuffer_direct_custom_op_allowlist=args.flatbuffer_direct_custom_op_allowlist,
        metadata=metadata,
    )
    float32_path = Path(payload["float32_tflite"])
    print("Conversion complete:")
    print(f"  ONNX    {payload['onnx_path']} ({format_size(Path(payload['onnx_path']))})")
    print(f"  FP32    {float32_path} ({format_size(float32_path)})")
    if payload["removed_generated_float16_tflite"]:
        print(f"  Removed generated FP16 TFLite: {payload['removed_generated_float16_tflite_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

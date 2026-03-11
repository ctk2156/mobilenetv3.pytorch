#!/usr/bin/env python3
"""Export optimized ShuffleNetV2 with custom ChannelShuffleSplit to fp32 TFLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper, shape_inference

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shufflenetv2 import shufflenetv2 as shufflenetv2_baseline
from shufflenetv2_custom import (
    shufflenetv2_custom_channel_shuffle,
)
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
    return f"shufflenetv2_{args.model_size}_custom_cs"


def transfer_state_dict_by_order(
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
) -> None:
    target_state = target_model.state_dict()
    source_state = source_model.state_dict()
    if len(target_state) != len(source_state):
        raise RuntimeError(
            f"State dict size mismatch: target={len(target_state)} source={len(source_state)}"
        )

    remapped = {}
    for (target_key, target_tensor), (source_key, source_tensor) in zip(
        target_state.items(),
        source_state.items(),
    ):
        if tuple(target_tensor.shape) != tuple(source_tensor.shape):
            raise RuntimeError(
                "Tensor shape mismatch while remapping ShuffleNetV2 weights: "
                f"{target_key} {tuple(target_tensor.shape)} vs "
                f"{source_key} {tuple(source_tensor.shape)}"
            )
        remapped[target_key] = source_tensor.detach().clone()
    target_model.load_state_dict(remapped)


def validate_equivalence(
    baseline_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    image_size: int,
) -> float:
    with torch.no_grad():
        sample = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
        baseline_output = baseline_model(sample)
        optimized_output = optimized_model(sample)
    return float((baseline_output - optimized_output).abs().max().item())


def _extract_shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    shape_map: dict[str, list[int]] = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                dims = []
                break
            dims.append(int(dim.dim_value))
        if dims:
            shape_map[value_info.name] = dims
    for initializer in model.graph.initializer:
        dims = [int(dim) for dim in initializer.dims]
        if dims:
            shape_map[initializer.name] = dims
    return shape_map


def _upsert_value_info(
    model: onnx.ModelProto,
    name: str,
    shape: list[int],
    elem_type: int = TensorProto.FLOAT,
) -> None:
    graph = model.graph
    for value_info in graph.value_info:
        if value_info.name != name:
            continue
        new_value_info = helper.make_tensor_value_info(name, elem_type, shape)
        value_info.CopyFrom(new_value_info)
        return
    graph.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))


def repair_channel_shuffle_split_shapes(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path))
    model = shape_inference.infer_shapes(model)
    for _ in range(16):
        shape_map = _extract_shape_map(model)
        changed = False
        for node in model.graph.node:
            if node.op_type == "Transpose" and len(node.input) == 1 and len(node.output) == 1:
                input_shape = shape_map.get(node.input[0], [])
                if len(input_shape) != 4:
                    continue
                perm = None
                for attr in node.attribute:
                    if attr.name == "perm":
                        perm = [int(v) for v in attr.ints]
                        break
                if perm is None or len(perm) != 4:
                    continue
                output_shape = [input_shape[index] for index in perm]
                if shape_map.get(node.output[0]) == output_shape:
                    continue
                _upsert_value_info(model, node.output[0], output_shape)
                shape_map[node.output[0]] = output_shape
                changed = True
                continue

            if node.op_type != "ChannelShuffleSplit" or len(node.input) != 2 or len(node.output) != 2:
                continue
            input_a_shape = shape_map.get(node.input[0], [])
            input_b_shape = shape_map.get(node.input[1], [])
            if len(input_a_shape) != 4 or len(input_b_shape) != 4:
                continue
            if (
                input_a_shape[0] != input_b_shape[0]
                or input_a_shape[1] != input_b_shape[1]
                or input_a_shape[2] != input_b_shape[2]
            ):
                continue
            total_channels = input_a_shape[3] + input_b_shape[3]
            if total_channels % 2 != 0:
                continue
            output_shape = [
                input_a_shape[0],
                input_a_shape[1],
                input_a_shape[2],
                total_channels // 2,
            ]
            for output_name in node.output:
                if shape_map.get(output_name) == output_shape:
                    continue
                _upsert_value_info(model, output_name, output_shape)
                shape_map[output_name] = output_shape
                changed = True
        if not changed:
            break
    onnx.save(model, str(onnx_path))


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

    baseline_model = shufflenetv2_baseline(
        input_size=int(args.image_size),
        num_classes=int(args.num_classes),
        model_size=args.model_size,
    ).eval()
    optimized_model = shufflenetv2_custom_channel_shuffle(
        input_size=int(args.image_size),
        num_classes=int(args.num_classes),
        model_size=args.model_size,
    ).eval()
    transfer_state_dict_by_order(optimized_model, baseline_model)
    max_abs_diff = validate_equivalence(
        baseline_model,
        optimized_model,
        int(args.image_size),
    )

    example = torch.randn(1, 3, int(args.image_size), int(args.image_size), dtype=torch.float32)

    print(f"[1/2] Exporting ONNX -> {onnx_path}")
    torch.onnx.export(
        optimized_model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamo=False,
    )
    repair_channel_shuffle_split_shapes(onnx_path)

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
        flatbuffer_direct_allow_custom_ops=True,
        flatbuffer_direct_custom_op_allowlist=["ChannelShuffleSplit"],
        metadata={
            "baseline_model_impl": "official_megvii_shufflenetv2",
            "custom_op_name": "ONNX_CHANNELSHUFFLESPLIT",
            "model_impl": "official_megvii_shufflenetv2_custom_channel_shuffle",
            "model_size": args.model_size,
            "validation_max_abs_diff_vs_baseline": max_abs_diff,
            "weights": "random",
        },
    )

    print("Export complete:")
    print(f"  ONNX    {onnx_path} ({format_size(onnx_path)})")
    print(f"  FP32    {payload['float32_tflite']} ({format_size(Path(payload['float32_tflite']))})")
    print(f"  Max abs diff vs baseline ShuffleNetV2: {max_abs_diff:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

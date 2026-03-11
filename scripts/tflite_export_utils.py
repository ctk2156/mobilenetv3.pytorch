#!/usr/bin/env python3
"""Shared helpers for ONNX -> fp32 TFLite export."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX2TF_ROOT = REPO_ROOT / "onnx2tf"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_tflite_export")


def _ensure_lightweight_onnx2tf_packages() -> None:
    package_root = ONNX2TF_ROOT / "onnx2tf"
    package_specs = {
        "onnx2tf": package_root,
        "onnx2tf.utils": package_root / "utils",
        "onnx2tf.tflite_builder": package_root / "tflite_builder",
        "onnx2tf.tflite_builder.preprocess": package_root / "tflite_builder" / "preprocess",
    }
    for package_name, package_path in package_specs.items():
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__file__ = str(package_path / "__init__.py")
        module.__path__ = [str(package_path)]
        sys.modules[package_name] = module


def _load_flatbuffer_direct_modules() -> dict[str, Any]:
    _ensure_lightweight_onnx2tf_packages()
    return {
        "ir": importlib.import_module("onnx2tf.tflite_builder.ir"),
        "lower": importlib.import_module("onnx2tf.tflite_builder.lower_from_onnx2tf"),
        "model_writer": importlib.import_module("onnx2tf.tflite_builder.model_writer"),
        "pipeline": importlib.import_module("onnx2tf.tflite_builder.preprocess.pipeline"),
        "rules": importlib.import_module("onnx2tf.tflite_builder.preprocess.rules"),
        "schema_loader": importlib.import_module("onnx2tf.tflite_builder.schema_loader"),
    }


def format_size(path: Path) -> str:
    size = path.stat().st_size
    mib = size / (1024.0 * 1024.0)
    return f"{mib:.2f} MiB"


def prepare_test_data(
    *,
    artifacts_root: Path,
    input_height: int,
    input_width: int,
) -> Path:
    test_data_path = (
        artifacts_root / "tflite" / f"test_data_nhwc_{input_height}x{input_width}.npy"
    )
    if test_data_path.exists():
        return test_data_path

    test_data_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=0)
    sample = rng.random((1, input_height, input_width, 3), dtype=np.float32)
    np.save(test_data_path, sample)
    return test_data_path


def _const_int_list(model_ir: Any, tensor_name: str) -> list[int] | None:
    tensor = model_ir.tensors.get(tensor_name)
    if tensor is None or tensor.data is None:
        return None
    values = np.asarray(tensor.data)
    if values.size == 0:
        return None
    return [int(v) for v in values.reshape(-1).tolist()]


def _set_tensor_shape(model_ir: Any, tensor_name: str, shape: list[int]) -> bool:
    tensor = model_ir.tensors.get(tensor_name)
    if tensor is None:
        return False
    normalized = [int(v) for v in shape]
    signature = list(normalized)
    changed = list(getattr(tensor, "shape", [])) != normalized or list(
        getattr(tensor, "shape_signature", normalized)
    ) != signature
    tensor.shape = normalized
    tensor.shape_signature = signature
    return changed


def _infer_spatial_out_dim(
    input_size: int,
    stride: int,
    dilation: int,
    kernel: int,
    padding: str,
) -> int:
    effective_kernel = (kernel - 1) * dilation + 1
    if padding == "SAME":
        return int((input_size + stride - 1) // stride)
    return int((input_size - effective_kernel + stride) // stride)


def _repair_model_ir_shapes(model_ir: Any) -> None:
    for _ in range(32):
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            if op_type == "PAD" and len(op.inputs) >= 2 and len(op.outputs) == 1:
                input_tensor = model_ir.tensors.get(op.inputs[0])
                paddings = _const_int_list(model_ir, op.inputs[1])
                if input_tensor is None or paddings is None:
                    continue
                input_shape = list(input_tensor.shape)
                if len(input_shape) != 4 or len(paddings) != 8:
                    continue
                output_shape = [
                    input_shape[dim] + paddings[dim * 2] + paddings[dim * 2 + 1]
                    for dim in range(4)
                ]
                changed = _set_tensor_shape(model_ir, op.outputs[0], output_shape) or changed
                continue

            if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"} and len(op.inputs) >= 2 and len(op.outputs) == 1:
                input_tensor = model_ir.tensors.get(op.inputs[0])
                filter_tensor = model_ir.tensors.get(op.inputs[1])
                if input_tensor is None or filter_tensor is None:
                    continue
                input_shape = list(input_tensor.shape)
                filter_shape = list(filter_tensor.shape)
                if len(input_shape) != 4 or len(filter_shape) != 4:
                    continue
                options = dict(op.options)
                stride_h = int(options.get("strideH", 1))
                stride_w = int(options.get("strideW", 1))
                dilation_h = int(options.get("dilationHFactor", 1))
                dilation_w = int(options.get("dilationWFactor", 1))
                padding = str(options.get("padding", "VALID")).upper()
                kernel_h = int(filter_shape[1])
                kernel_w = int(filter_shape[2])
                output_channels = int(filter_shape[0] if op_type == "CONV_2D" else filter_shape[3])
                output_shape = [
                    input_shape[0],
                    _infer_spatial_out_dim(input_shape[1], stride_h, dilation_h, kernel_h, padding),
                    _infer_spatial_out_dim(input_shape[2], stride_w, dilation_w, kernel_w, padding),
                    output_channels,
                ]
                changed = _set_tensor_shape(model_ir, op.outputs[0], output_shape) or changed
                continue

            if op_type in {"MAX_POOL_2D", "AVERAGE_POOL_2D"} and len(op.inputs) >= 1 and len(op.outputs) == 1:
                input_tensor = model_ir.tensors.get(op.inputs[0])
                if input_tensor is None:
                    continue
                input_shape = list(input_tensor.shape)
                if len(input_shape) != 4:
                    continue
                options = dict(op.options)
                stride_h = int(options.get("strideH", 1))
                stride_w = int(options.get("strideW", 1))
                filter_h = int(options.get("filterHeight", 1))
                filter_w = int(options.get("filterWidth", 1))
                padding = str(options.get("padding", "VALID")).upper()
                output_shape = [
                    input_shape[0],
                    _infer_spatial_out_dim(input_shape[1], stride_h, 1, filter_h, padding),
                    _infer_spatial_out_dim(input_shape[2], stride_w, 1, filter_w, padding),
                    input_shape[3],
                ]
                changed = _set_tensor_shape(model_ir, op.outputs[0], output_shape) or changed
                continue

            if op_type == "CONCATENATION" and len(op.outputs) == 1:
                options = dict(op.options)
                axis = int(options.get("axis", 0))
                input_shapes = []
                for name in op.inputs:
                    tensor = model_ir.tensors.get(name)
                    if tensor is None:
                        input_shapes = []
                        break
                    shape = list(tensor.shape)
                    if len(shape) != 4:
                        input_shapes = []
                        break
                    input_shapes.append(shape)
                if not input_shapes:
                    continue
                output_shape = list(input_shapes[0])
                output_shape[axis] = sum(shape[axis] for shape in input_shapes)
                changed = _set_tensor_shape(model_ir, op.outputs[0], output_shape) or changed
                continue

            if op_type == "RESHAPE" and len(op.inputs) >= 1 and len(op.outputs) == 1:
                input_tensor = model_ir.tensors.get(op.inputs[0])
                if input_tensor is None:
                    continue
                input_shape = list(input_tensor.shape)
                new_shape = [int(v) for v in op.options.get("newShape", [])]
                if not new_shape:
                    continue
                if -1 in new_shape:
                    known_product = 1
                    unknown_index = -1
                    for idx, dim in enumerate(new_shape):
                        if dim == -1:
                            unknown_index = idx
                        else:
                            known_product *= dim
                    input_product = int(np.prod(input_shape))
                    if unknown_index >= 0 and known_product != 0:
                        new_shape[unknown_index] = input_product // known_product
                changed = _set_tensor_shape(model_ir, op.outputs[0], new_shape) or changed
                continue

            if op_type == "CUSTOM" and len(op.inputs) == 2 and len(op.outputs) == 2:
                options = dict(op.options)
                if str(options.get("customCode", "")) != "ONNX_CHANNELSHUFFLESPLIT":
                    continue
                input_a = model_ir.tensors.get(op.inputs[0])
                input_b = model_ir.tensors.get(op.inputs[1])
                if input_a is None or input_b is None:
                    continue
                shape_a = list(input_a.shape)
                shape_b = list(input_b.shape)
                if len(shape_a) != 4 or len(shape_b) != 4:
                    continue
                if shape_a[:3] != shape_b[:3]:
                    continue
                total_channels = int(shape_a[3]) + int(shape_b[3])
                if total_channels % 2 != 0:
                    continue
                output_shape = [shape_a[0], shape_a[1], shape_a[2], total_channels // 2]
                changed = _set_tensor_shape(model_ir, op.outputs[0], output_shape) or changed
                changed = _set_tensor_shape(model_ir, op.outputs[1], output_shape) or changed
                continue

        if not changed:
            break


def _optimize_padv2_pool_patterns(model_ir: Any) -> None:
    consumer_count: dict[str, int] = {}
    producer_index: dict[str, int] = {}
    for index, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producer_index[output_name] = index
        for input_name in op.inputs:
            consumer_count[input_name] = int(consumer_count.get(input_name, 0) + 1)

    remove_indices: set[int] = set()
    for index, op in enumerate(model_ir.operators):
        op_type = str(op.op_type)
        if op_type not in {"MAX_POOL_2D", "AVERAGE_POOL_2D"} or len(op.inputs) != 1:
            continue
        padded_input_name = str(op.inputs[0])
        pad_index = producer_index.get(padded_input_name)
        if pad_index is None:
            continue
        pad_op = model_ir.operators[pad_index]
        if str(pad_op.op_type) != "PADV2" or len(pad_op.inputs) < 3 or len(pad_op.outputs) != 1:
            continue
        if consumer_count.get(pad_op.outputs[0], 0) != 1:
            continue

        pad_input_name = str(pad_op.inputs[0])
        paddings = _const_int_list(model_ir, pad_op.inputs[1])
        pad_values = _const_int_list(model_ir, pad_op.inputs[2])
        pad_input_tensor = model_ir.tensors.get(pad_input_name)
        if pad_input_tensor is None or paddings is None or pad_values is None:
            continue
        input_shape = list(pad_input_tensor.shape)
        if len(input_shape) != 4 or len(paddings) != 8:
            continue

        batch_pad = paddings[0:2]
        height_pad = paddings[2:4]
        width_pad = paddings[4:6]
        channel_pad = paddings[6:8]
        if batch_pad != [0, 0] or channel_pad != [0, 0]:
            continue

        options = dict(op.options)
        filter_h = int(options.get("filterHeight", 1))
        filter_w = int(options.get("filterWidth", 1))
        stride_h = int(options.get("strideH", 1))
        stride_w = int(options.get("strideW", 1))

        if op_type == "MAX_POOL_2D":
            if height_pad[0] == height_pad[1] and width_pad[0] == width_pad[1]:
                op.inputs[0] = pad_input_name
                op.options = dict(op.options)
                op.options["padding"] = "SAME"
                remove_indices.add(pad_index)
        elif op_type == "AVERAGE_POOL_2D":
            if (
                pad_values == [0]
                and height_pad[0] == 0
                and width_pad[0] == 0
                and input_shape[1] == filter_h
                and input_shape[2] == filter_w
                and stride_h == filter_h
                and stride_w == filter_w
            ):
                op.inputs[0] = pad_input_name
                remove_indices.add(pad_index)

    if remove_indices:
        model_ir.operators = [
            op for idx, op in enumerate(model_ir.operators) if idx not in remove_indices
        ]


def convert_onnx_to_fp32_tflite(
    *,
    onnx_path: Path,
    output_name: str,
    input_height: int,
    input_width: int,
    artifacts_root: Path,
    force: bool,
    enable_accumulation_type_float16: bool,
    output_signaturedefs: bool,
    flatbuffer_direct_allow_custom_ops: bool = False,
    flatbuffer_direct_custom_op_allowlist: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    onnx_path = onnx_path.resolve()
    artifacts_root = artifacts_root.resolve()
    tflite_dir = artifacts_root / "tflite" / output_name
    tflite_dir.parent.mkdir(parents=True, exist_ok=True)

    if force:
        if tflite_dir.exists():
            shutil.rmtree(tflite_dir)
    elif tflite_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing TFLite directory: {tflite_dir}")

    test_data_path = prepare_test_data(
        artifacts_root=artifacts_root,
        input_height=input_height,
        input_width=input_width,
    )
    modules = _load_flatbuffer_direct_modules()
    pipeline = modules["pipeline"]
    rules = modules["rules"]
    lower = modules["lower"]
    ir = modules["ir"]
    model_writer = modules["model_writer"]
    schema_loader = modules["schema_loader"]

    onnx_graph = onnx.load(str(onnx_path))
    pipeline.clear_preprocess_rules()
    rules.register_default_preprocess_rules()
    preprocessed_onnx_graph, _ = pipeline.run_preprocess_pipeline(
        onnx_graph=onnx_graph
    )
    model_ir = lower.lower_onnx_to_ir(
        onnx_graph=preprocessed_onnx_graph,
        output_file_name=output_name,
        allow_custom_ops=bool(flatbuffer_direct_allow_custom_ops),
        custom_op_allowlist=flatbuffer_direct_custom_op_allowlist,
        transpose_inputs_to_nhwc=True,
        show_progress=False,
    )
    _repair_model_ir_shapes(model_ir)
    _optimize_padv2_pool_patterns(model_ir)
    _repair_model_ir_shapes(model_ir)

    model_ir_fp32 = ir.clone_model_ir_with_float32(model_ir)
    ir.prune_identity_cast_operators(
        model_ir_fp32,
        preserve_model_outputs=True,
    )
    ir.optimize_redundant_transpose_operators(
        model_ir_fp32,
        preserve_model_outputs=True,
    )

    schema_tflite = schema_loader.load_schema_module(str(tflite_dir))

    float32_path = tflite_dir / f"{output_name}_float32.tflite"
    model_writer.write_model_file(
        schema_tflite=schema_tflite,
        model_ir=model_ir_fp32,
        output_tflite_path=str(float32_path),
    )

    tensor_correspondence_report = lower.build_tensor_correspondence_report(
        onnx_graph=preprocessed_onnx_graph,
        model_ir=model_ir_fp32,
    )
    lower.write_tensor_correspondence_report(
        report=tensor_correspondence_report,
        output_report_path=str(
            tflite_dir / f"{output_name}_tensor_correspondence_report.json"
        ),
    )

    float16_path = tflite_dir / f"{output_name}_float16.tflite"
    removed_generated_float16 = False
    if float16_path.exists():
        float16_path.unlink()
        removed_generated_float16 = True

    payload: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "float32_tflite": str(float32_path),
        "input_shape_nhwc": [1, input_height, input_width, 3],
        "runtime_precision_policy": (
            "Retain float32 TFLite only. Use XNNPACK runtime flags to switch "
            "between fp32 and fp16 execution."
        ),
        "removed_generated_float16_tflite": removed_generated_float16,
        "removed_generated_float16_tflite_path": (
            str(float16_path) if removed_generated_float16 else ""
        ),
        "test_data_nhwc_path": str(test_data_path),
        "enable_accumulation_type_float16": bool(enable_accumulation_type_float16),
        "output_signaturedefs": bool(output_signaturedefs),
        "flatbuffer_direct_allow_custom_ops": bool(
            flatbuffer_direct_allow_custom_ops
        ),
        "flatbuffer_direct_custom_op_allowlist": list(
            flatbuffer_direct_custom_op_allowlist or []
        ),
    }
    if metadata:
        payload.update(metadata)

    (tflite_dir / "export_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload

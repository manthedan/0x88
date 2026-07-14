"""Rewrite large LC0 f32 MatMuls to dynamic uint8/int8 MatMulInteger.

Each selected activation is quantized with DynamicQuantizeLinear. Constant
weights use symmetric per-output-column int8 quantization. MatMulInteger
accumulates to int32, then the result is cast and rescaled to f32 so graph
inputs, residual paths, normalization, heads, and outputs remain floating
point.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--min-elements", type=int, default=4096)
    parser.add_argument("--include-name-regex")
    parser.add_argument("--exclude-name-regex")
    parser.add_argument("--keep-smolgen-f32", action="store_true")
    parser.add_argument("--max-matmuls", type=int)
    parser.add_argument("--report")
    return parser.parse_args()


def tensor_bytes(initializer: onnx.TensorProto) -> int:
    return int(numpy_helper.to_array(initializer).nbytes)


def inferred_shapes(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    inferred = shape_inference.infer_shapes(model, data_prop=False)
    shapes: dict[str, list[int | None]] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shapes[value.name] = [
            int(dimension.dim_value) if dimension.HasField("dim_value") else None
            for dimension in tensor_type.shape.dim
        ]
    return shapes


def matmul_macs(activation_shape: list[int | None] | None, weight_shape: tuple[int, ...]) -> int | None:
    if not activation_shape or len(activation_shape) < 1 or len(weight_shape) != 2:
        return None
    leading = activation_shape[:-1]
    if any(dimension is None for dimension in leading):
        return None
    return int(np.prod(leading, dtype=np.int64)) * int(weight_shape[0]) * int(weight_shape[1])


def main() -> None:
    args = parse_args()
    model = onnx.load(args.src)
    shapes = inferred_shapes(model)
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    consumers: dict[str, list[int]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node_index)

    include = re.compile(args.include_name_regex) if args.include_name_regex else None
    exclude_patterns = [args.exclude_name_regex] if args.exclude_name_regex else []
    if args.keep_smolgen_f32:
        exclude_patterns.append(r"/smolgen/|/const/smolgen_w")
    exclude = re.compile("|".join(f"(?:{pattern})" for pattern in exclude_patterns)) if exclude_patterns else None
    targets: dict[int, tuple[onnx.TensorProto, np.ndarray]] = {}
    skipped = Counter()
    for node_index, node in enumerate(graph.node):
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        weight = initializers.get(node.input[1])
        if weight is None:
            skipped["dynamic_weight"] += 1
            continue
        array = numpy_helper.to_array(weight)
        identity = f"{node.name} {weight.name}"
        if include and not include.search(identity):
            skipped["include_filter"] += 1
            continue
        if exclude and exclude.search(identity):
            skipped["exclude_filter"] += 1
            continue
        if array.dtype != np.float32:
            skipped["non_f32_weight"] += 1
            continue
        if array.ndim != 2:
            skipped["non_matrix_weight"] += 1
            continue
        if array.size < args.min_elements:
            skipped["below_min_elements"] += 1
            continue
        if args.max_matmuls is not None and len(targets) >= max(0, args.max_matmuls):
            skipped["max_matmuls"] += 1
            continue
        targets[node_index] = (weight, array)

    quantized_weights: dict[str, tuple[str, str, str, dict[str, object]]] = {}
    replacement_initializers: list[onnx.TensorProto] = []
    target_indices = set(targets)
    removable_weights: set[str] = set()
    for weight, array in targets.values():
        if weight.name in quantized_weights:
            continue
        values = array.astype(np.float32, copy=False)
        max_abs = np.max(np.abs(values), axis=0)
        scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
        quantized = np.clip(np.rint(values / scale[None, :]), -127, 127).astype(np.int8)
        dequantized = quantized.astype(np.float32) * scale[None, :]
        error = dequantized - values
        denominator = float(np.sqrt(np.mean(values * values))) or 1.0
        q_name = f"{weight.name}__mmi_q8"
        scale_name = f"{weight.name}__mmi_scale"
        zero_name = f"{weight.name}__mmi_zero"
        replacement_initializers.extend([
            numpy_helper.from_array(quantized, q_name),
            numpy_helper.from_array(scale, scale_name),
            numpy_helper.from_array(np.array(0, dtype=np.int8), zero_name),
        ])
        weight_report = {
            "name": weight.name,
            "shape": list(array.shape),
            "elements": int(array.size),
            "sourceBytes": int(array.nbytes),
            "quantizedBytes": int(quantized.nbytes + scale.nbytes + 1),
            "maxAbsError": float(np.max(np.abs(error))),
            "relativeRmsError": float(np.sqrt(np.mean(error * error)) / denominator),
        }
        quantized_weights[weight.name] = (q_name, scale_name, zero_name, weight_report)
        if all(consumer_index in target_indices for consumer_index in consumers[weight.name]):
            removable_weights.add(weight.name)

    new_nodes: list[onnx.NodeProto] = []
    activation_quantizers: dict[str, tuple[str, str, str]] = {}
    matmul_rows: list[dict[str, object]] = []
    for node_index, node in enumerate(graph.node):
        target = targets.get(node_index)
        if target is None:
            new_nodes.append(node)
            continue
        weight, array = target
        activation_name = node.input[0]
        activation_outputs = activation_quantizers.get(activation_name)
        if activation_outputs is None:
            activation_outputs = (
                f"{activation_name}__mmi_u8",
                f"{activation_name}__mmi_scale",
                f"{activation_name}__mmi_zero",
            )
            activation_quantizers[activation_name] = activation_outputs
            new_nodes.append(helper.make_node(
                "DynamicQuantizeLinear",
                [activation_name],
                list(activation_outputs),
                name=f"DynamicQuantizeLinear_{len(activation_quantizers) - 1}",
            ))
        activation_q, activation_scale, activation_zero = activation_outputs
        weight_q, weight_scale, weight_zero, _ = quantized_weights[weight.name]
        output_name = node.output[0]
        integer_output = f"{output_name}__mmi_i32"
        float_output = f"{output_name}__mmi_f32"
        combined_scale = f"{output_name}__mmi_combined_scale"
        base_name = node.name or f"MatMul_{node_index}"
        new_nodes.extend([
            helper.make_node(
                "MatMulInteger",
                [activation_q, weight_q, activation_zero, weight_zero],
                [integer_output],
                name=f"{base_name}__integer",
            ),
            helper.make_node(
                "Cast",
                [integer_output],
                [float_output],
                name=f"{base_name}__cast",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Mul",
                [activation_scale, weight_scale],
                [combined_scale],
                name=f"{base_name}__scale",
            ),
            helper.make_node(
                "Mul",
                [float_output, combined_scale],
                [output_name],
                name=f"{base_name}__dequantize",
            ),
        ])
        matmul_rows.append({
            "nodeIndex": node_index,
            "nodeName": node.name,
            "activation": activation_name,
            "weight": weight.name,
            "shape": list(array.shape),
            "activationShape": shapes.get(activation_name),
            "macsPerInference": matmul_macs(shapes.get(activation_name), array.shape),
        })

    retained_initializers = [
        initializer for initializer in graph.initializer
        if initializer.name not in removable_weights
    ]
    del graph.initializer[:]
    graph.initializer.extend(retained_initializers)
    graph.initializer.extend(replacement_initializers)
    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.checker.check_model(model, full_check=True)
    destination = Path(args.dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)

    source_initializer_bytes = sum(tensor_bytes(initializer) for initializer in initializers.values())
    output_initializer_bytes = sum(tensor_bytes(initializer) for initializer in graph.initializer)
    operator_counts = Counter(node.op_type for node in graph.node)
    summary = {
        "schema": "lc0_browser.onnx_dynamic_mmi_int8.v1",
        "source": str(args.src),
        "output": str(destination),
        "activationQuantization": "dynamic-tensor",
        "keepSmolgenF32": args.keep_smolgen_f32,
        "selectedMatMuls": len(matmul_rows),
        "uniqueQuantizedWeights": len(quantized_weights),
        "sharedActivationQuantizers": len(activation_quantizers),
        "selectedMacsPerInference": (
            sum(row["macsPerInference"] for row in matmul_rows)
            if all(row["macsPerInference"] is not None for row in matmul_rows)
            else None
        ),
        "removedFloatWeights": len(removable_weights),
        "sourceInitializerBytes": source_initializer_bytes,
        "outputInitializerBytes": output_initializer_bytes,
        "initializerByteRatio": output_initializer_bytes / source_initializer_bytes if source_initializer_bytes else None,
        "outputBytes": destination.stat().st_size,
        "operatorCounts": dict(sorted(operator_counts.items())),
        "skippedMatMuls": dict(sorted(skipped.items())),
    }
    report_path = Path(args.report) if args.report else destination.with_suffix(".mmi-report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "summary": summary,
        "weights": [value[3] for value in quantized_weights.values()],
        "matmuls": matmul_rows,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

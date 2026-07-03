#!/usr/bin/env python3
"""
Quantize an ONNX reranker model using dynamic INT8 quantization.

Usage:
  RERANKER_ONNX_PATH=path/to/model.onnx python scripts/quantize_reranker.py
  # optional envs:
  #   OUTPUT_ONNX_PATH=path/to/output.onnx (default: <input>_int8.onnx)

Notes:
- Uses onnxruntime.quantization.quantize_dynamic
- Keeps model operators compatible with CPUExecutionProvider by default
- Safe no-op if inputs are missing or quantization fails; prints error
"""
import os
import sys


def main():
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore
    except Exception:
        print("ERROR: onnxruntime with quantization extras is required.")
        sys.exit(2)

    in_path = os.environ.get("RERANKER_ONNX_PATH", "").strip()
    if not in_path or not os.path.exists(in_path):
        print("ERROR: set RERANKER_ONNX_PATH to an existing .onnx file")
        sys.exit(2)

    out_path = os.environ.get("OUTPUT_ONNX_PATH", "").strip() or (
        in_path[:-5] + "_int8.onnx"
        if in_path.endswith(".onnx")
        else in_path + ".int8.onnx"
    )

    try:
        quantize_dynamic(
            model_input=in_path,
            model_output=out_path,
            weight_type=QuantType.QUInt8,
            optimize_model=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
            },
        )
        print(f"Quantized model saved to {out_path}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

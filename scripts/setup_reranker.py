#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]


def upsert_env_line(path: Path, key: str, value: str):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    replaced = False
    new_lines = []
    for ln in lines:
        if ln.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(ln)
    if not replaced:
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")
    path.write_text("\n".join(new_lines) + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Download ONNX cross-encoder + tokenizer and set .env paths"
    )
    p.add_argument("--onnx-url", required=True, help="Direct URL to .onnx model file")
    p.add_argument(
        "--tokenizer-url", required=True, help="Direct URL to tokenizer.json file"
    )
    p.add_argument(
        "--dest",
        default="models",
        help="Destination directory under repo root (default: models)",
    )
    p.add_argument(
        "--onnx-filename", default=None, help="Optional override for ONNX filename"
    )
    p.add_argument(
        "--tokenizer-filename",
        default=None,
        help="Optional override for tokenizer filename",
    )
    args = p.parse_args()

    dest_dir = ROOT / args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)

    onnx_name = args.onnx_filename or Path(args.onnx_url).name or "reranker.onnx"
    tok_name = (
        args.tokenizer_filename or Path(args.tokenizer_url).name or "tokenizer.json"
    )

    onnx_path = dest_dir / onnx_name
    tok_path = dest_dir / tok_name

    print(f"Downloading ONNX model to {onnx_path} ...")
    urlretrieve(args.onnx_url, onnx_path)
    print(f"Downloading tokenizer to {tok_path} ...")
    urlretrieve(args.tokenizer_url, tok_path)

    # Paths inside the container map to /work
    container_onnx = f"/work/{args.dest.rstrip('/')}/{onnx_name}"
    container_tok = f"/work/{args.dest.rstrip('/')}/{tok_name}"

    # Update .env and .env.example
    env_path = ROOT / ".env"
    env_example = ROOT / ".env.example"
    upsert_env_line(env_path, "RERANKER_ONNX_PATH", container_onnx)
    upsert_env_line(env_path, "RERANKER_TOKENIZER_PATH", container_tok)
    upsert_env_line(env_example, "RERANKER_ONNX_PATH", container_onnx)
    upsert_env_line(env_example, "RERANKER_TOKENIZER_PATH", container_tok)

    print("\nUpdated .env and .env.example:")
    print(f"  RERANKER_ONNX_PATH={container_onnx}")
    print(f"  RERANKER_TOKENIZER_PATH={container_tok}")
    print("\nNext steps:")
    print("  - Run: make rerank-local")


if __name__ == "__main__":
    main()

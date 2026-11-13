#!/usr/bin/env python3
"""Export the trained amu checkpoint to ONNX so it can run in a browser.

This script only *reads* the main application (``../tinymind-ai``): it imports
``model.py``, loads ``checkpoints/amu.pt`` and writes three files into
``amu-web/model/``. Nothing in the main app is modified.

    python export_onnx.py                     # uses ../tinymind-ai/checkpoints/amu.pt
    python export_onnx.py --checkpoint ../tinymind-ai/checkpoints/amu-final.pt

Output:
    model/amu.onnx     the network, batch and sequence length dynamic
    model/vocab.json   the ordered character list (the tokenizer)
    model/config.json  block_size, vocab_size, parameter count, source hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_APP = (HERE.parent / "tinymind-ai").resolve()

DEFAULT_CHECKPOINT = MAIN_APP / "checkpoints" / "amu.pt"
OUT_DIR = HERE / "model"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seq", type=int, default=16, help="dummy sequence length")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out).resolve()

    if not checkpoint.exists():
        raise SystemExit("no checkpoint at %s (train it first: python ../tinymind-ai/train.py)" % checkpoint)

    import torch

    sys.path.insert(0, str(MAIN_APP))
    from model import Amu  # noqa: E402  (path is set above on purpose)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    characters = list(payload["vocab"])

    model = Amu(**config)
    model.load_state_dict(payload["model_state"])
    model.eval()

    block_size = int(model.block_size)
    vocab_size = int(model.vocab_size)
    parameters = int(sum(p.numel() for p in model.parameters()))

    if len(characters) != vocab_size:
        raise SystemExit(
            "vocabulary has %d characters but the model expects %d"
            % (len(characters), vocab_size)
        )

    seq = max(2, min(args.seq, block_size))
    dummy = torch.zeros((1, seq), dtype=torch.long)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "amu.onnx"

    # Full-sequence forward pass: the browser feeds the whole context each step
    # and keeps the last `block_size` characters as its sliding window. That
    # avoids exporting the key/value cache, which keeps the graph simple and
    # portable across onnxruntime-web backends.
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={"tokens": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    (out_dir / "vocab.json").write_text(
        json.dumps(characters, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "model": "amu",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_of(checkpoint),
        "parameters": parameters,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "embed_dim": int(config.get("embed_dim", 0)),
        "heads": int(config.get("heads", 0)),
        "kv_heads": int(config.get("kv_heads") or config.get("heads", 0)),
        "layers": int(config.get("layers", 0)),
        "opset": args.opset,
        "onnx_bytes": onnx_path.stat().st_size,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("wrote %s (%d bytes)" % (onnx_path, meta["onnx_bytes"]))
    print("wrote %s (%d characters)" % (out_dir / "vocab.json", vocab_size))
    print("wrote %s" % (out_dir / "config.json"))
    print("parameters: %d  block_size: %d  opset: %d" % (parameters, block_size, args.opset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

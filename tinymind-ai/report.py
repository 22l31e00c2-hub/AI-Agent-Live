"""Print amu's report card: what was trained, how it scored, how fast it runs.

    python report.py               # best-validation checkpoint
    python report.py --final       # last state of the run
    python report.py --no-bench    # skip the speed measurement
    python report.py checkpoints/amu.pt

The card is read from the JSON sidecar that train.py writes next to every
checkpoint (checkpoints/amu.json), so it works without loading the model. The
optional speed line needs torch and a real checkpoint; without them the rest of
the card is still printed.
"""

import argparse
import json
import os
import time

DEFAULT_CHECKPOINT = "checkpoints/amu.pt"


def meta_path(checkpoint):
    """checkpoints/amu.pt -> checkpoints/amu.json"""
    return os.path.splitext(checkpoint)[0] + ".json"


def load_meta(checkpoint):
    with open(meta_path(checkpoint), "r", encoding="utf-8") as handle:
        return json.load(handle)


def human_bytes(size):
    if size < 1024:
        return "%d B" % size

    size = float(size)
    for unit in ("KB", "MB", "GB"):
        size /= 1024.0
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)


def _value(mapping, key, default="unknown"):
    value = mapping.get(key)
    return default if value is None else value


def _line(label, value):
    return "%-11s: %s" % (label, value)


def _number(value, pattern):
    try:
        return pattern % float(value)
    except (TypeError, ValueError):
        return str(value)


def _architecture(model):
    if not model:
        return None

    return (
        "%s layers, width %s, %s heads (%s kv), block %s, vocab %s; "
        "RMSNorm + RoPE + SwiGLU, tied embeddings"
        % (
            _value(model, "layers"),
            _value(model, "embed_dim"),
            _value(model, "heads"),
            _value(model, "kv_heads"),
            _value(model, "block_size"),
            _value(model, "vocab_size"),
        )
    )


def build_report(meta, checkpoint, size_bytes=None, benchmark=None):
    """Return the report as text. Pure function: no file or torch access."""
    model = meta.get("model") or {}
    training = meta.get("training") or {}
    corpus = meta.get("corpus") or {}
    losses = meta.get("losses") or {}

    lines = [
        "amu report card",
        _line("checkpoint", "%s (%s)" % (checkpoint, _value(meta, "kind"))),
    ]

    if size_bytes is not None:
        lines.append(_line("file", human_bytes(size_bytes)))

    if meta.get("created"):
        lines.append(_line("created", meta["created"]))

    if meta.get("parameters"):
        lines.append(_line("parameters", "{:,}".format(int(meta["parameters"]))))

    architecture = _architecture(model)
    if architecture:
        lines.append(_line("model", architecture))

    if corpus:
        lines.append(_line(
            "corpus",
            "%s characters (%s train / %s val), %s distinct characters"
            % (
                "{:,}".format(int(_value(corpus, "characters", 0))),
                "{:,}".format(int(_value(corpus, "train", 0))),
                "{:,}".format(int(_value(corpus, "validation", 0))),
                _value(corpus, "vocab_size"),
            ),
        ))
        if corpus.get("sha256"):
            lines.append(_line("checksum", "%s (%s)" % (
                corpus["sha256"][:16], _value(corpus, "source")
            )))

    if training:
        lines.append(_line(
            "schedule",
            "%s steps planned, lr %s, warmup %s, batch %s"
            % (
                _value(training, "steps_planned", meta.get("steps_planned")),
                _number(_value(training, "learning_rate"), "%.1e"),
                _value(training, "warmup_steps"),
                _value(training, "batch_size"),
            ),
        ))
        lines.append(_line(
            "reproduce",
            "seed %s, patience %s, eval every %s"
            % (
                _value(training, "seed"),
                _value(training, "patience"),
                _value(training, "eval_every"),
            ),
        ))

    if losses:
        lines.append(_line(
            "result",
            "%s step %s: train %s, val %s, %s bits/char"
            % (
                _value(meta, "kind", "checkpoint"),
                _value(meta, "step"),
                _number(_value(losses, "train", "?"), "%.4f"),
                _number(_value(losses, "val", "?"), "%.4f"),
                _number(_value(losses, "val_bits_per_char", "?"), "%.3f"),
            ),
        ))

    if meta.get("best_step") is not None and meta.get("kind") == "final":
        lines.append(_line(
            "best step",
            "%s%s"
            % (
                meta["best_step"],
                " (run stopped early)" if meta.get("early_stopped") else "",
            ),
        ))

    if benchmark:
        lines.append(_line(
            "speed",
            "%d characters/s (%d characters in %.2f s)"
            % (
                round(benchmark["chars_per_second"]),
                benchmark["characters"],
                benchmark["seconds"],
            ),
        ))
        if benchmark.get("sample"):
            lines.append(_line("sample", benchmark["sample"].replace("\n", "\n            ")))
    elif benchmark is None:
        lines.append(_line("speed", "not measured"))

    lines.append(_line("device", _value(meta, "device")))
    if meta.get("seconds") is not None:
        lines.append(_line("train time", "%s s" % meta["seconds"]))

    return "\n".join(lines)


def benchmark(checkpoint, characters=200, prompt="AutoCAD "):
    """Load a checkpoint and measure generation speed, including a sample."""
    import torch

    from model import Amu

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(checkpoint, map_location=device, weights_only=False)

    model = Amu(**data["config"]).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()

    vocab = data["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    tokens = [stoi[ch] for ch in prompt if ch in stoi] or [0]
    context = torch.tensor([tokens], dtype=torch.long, device=device)

    started = time.perf_counter()
    output = model.generate(
        context,
        max_new_tokens=characters,
        temperature=0.8,
        top_k=40,
    )
    seconds = max(time.perf_counter() - started, 1e-9)

    text = "".join(itos[int(i)] for i in output[0].tolist())

    return {
        "characters": characters,
        "seconds": seconds,
        "chars_per_second": characters / seconds,
        "sample": text,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print amu's report card and, optionally, measure speed."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=DEFAULT_CHECKPOINT,
        help="checkpoint to describe (default: %s)" % DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="use checkpoints/amu-final.pt instead",
    )
    parser.add_argument(
        "--no-bench",
        action="store_true",
        help="skip loading the model and measuring generation speed",
    )
    args = parser.parse_args(argv)

    checkpoint = "checkpoints/amu-final.pt" if args.final else args.checkpoint

    if not os.path.exists(checkpoint):
        raise SystemExit("No checkpoint at %s yet. Run: python train.py" % checkpoint)

    record = meta_path(checkpoint)
    if not os.path.exists(record):
        raise SystemExit(
            "No training record at %s. Retrain with: python train.py" % record
        )

    meta = load_meta(checkpoint)

    result = None
    if not args.no_bench:
        try:
            result = benchmark(checkpoint)
        except ImportError:
            result = None

    print(build_report(meta, checkpoint, os.path.getsize(checkpoint), result))

    if result is None and not args.no_bench:
        print("speed      : unavailable (torch is not installed in this interpreter)")


if __name__ == "__main__":
    main()

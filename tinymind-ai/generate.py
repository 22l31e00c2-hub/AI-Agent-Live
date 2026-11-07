"""Sample text from your trained amu.

Interactive:  python generate.py
One shot:     python generate.py "LAYER " 200
Last state:   python generate.py --final "LAYER " 200
Benchmark:    python generate.py --bench

`checkpoints/amu.pt` holds the best-validation weights. `amu-final.pt` is the
last state of the run, which on a small corpus has memorised the text instead
of generalising - useful to look at, not the model to quote.
"""

import os
import sys

import torch

from model import Amu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model

CHECKPOINT = (
    "checkpoints/amu-final.pt" if "--final" in sys.argv else "checkpoints/amu.pt"
)

if not os.path.exists(CHECKPOINT):
    raise SystemExit(
        "No checkpoint at %s yet. Run: python train.py" % CHECKPOINT
    )

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint["config"]
characters = checkpoint["vocab"]

stoi = {ch: i for i, ch in enumerate(characters)}
itos = {i: ch for i, ch in enumerate(characters)}

model = Amu(**config).to(DEVICE)

model.load_state_dict(checkpoint["model_state"])
model.eval()


def encode(text):
    return [stoi[ch] for ch in text if ch in stoi]


def decode(tokens):
    return "".join(itos[int(i)] for i in tokens)


def sample(prompt, max_new_tokens=200, temperature=0.8):
    tokens = encode(prompt)

    if not tokens:
        return None

    context = torch.tensor(
        [tokens], dtype=torch.long, device=DEVICE
    )

    output = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    return decode(output[0].tolist())


flags = [value for value in sys.argv[1:] if value != "--final"]

if flags and flags[0] in ("--bench", "-b"):
    import time

    prompt = flags[1] if len(flags) > 1 else "LAYER "
    limit = int(flags[2]) if len(flags) > 2 else 200

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

    started = time.perf_counter()
    output = model.generate(context, max_new_tokens=limit, temperature=0.8)
    elapsed = max(time.perf_counter() - started, 1e-9)

    print("amu bench")
    print("params     : %d" % sum(p.numel() for p in model.parameters()))
    print("block size : %d" % model.block_size)
    print(
        "speed      : %d characters in %.2f s -> %.0f characters/s"
        % (limit, elapsed, limit / elapsed)
    )
    print("sample     : %r" % decode(output[0].tolist()))
    raise SystemExit(0)


if flags:
    prompt = flags[0]
    limit = int(flags[1]) if len(flags) > 1 else 200
    print(sample(prompt, max_new_tokens=limit))
    raise SystemExit(0)

print("amu is ready!")
print("Type exit to quit.")

while True:
    prompt = input("\nYou: ")

    if prompt.lower() == "exit":
        break

    generated = sample(prompt)

    if generated is None:
        print("AI: Please enter characters from my vocabulary.")
        continue

    print("\nAI:", generated)

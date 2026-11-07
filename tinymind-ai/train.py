"""Train amu from scratch on data/input.txt.

Run:  python train.py
Env overrides (handy for a quick smoke test):
      AMU_STEPS=50 python train.py
"""

import os
import torch
import numpy as np

from model import Amu

# -----------------------------
# CONFIGURATION
# -----------------------------

BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_STEPS = int(os.environ.get("AMU_STEPS", "5000"))
LEARNING_RATE = 3e-4

EMBED_DIM = 128
HEADS = 4
LAYERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Training device:", DEVICE)

# -----------------------------
# LOAD YOUR DATA
# -----------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

if len(text) < BLOCK_SIZE + 2:
    raise ValueError(
        "Training text is too short. Add more text to input.txt."
    )

# Character-level tokenizer

characters = sorted(list(set(text)))

vocab_size = len(characters)

stoi = {ch: i for i, ch in enumerate(characters)}
itos = {i: ch for i, ch in enumerate(characters)}


def encode(text):
    return [stoi[ch] for ch in text if ch in stoi]


def decode(tokens):
    return "".join(itos[int(i)] for i in tokens)


data = torch.tensor(encode(text), dtype=torch.long)

# Train / validation split

split = int(0.9 * len(data))

train_data = data[:split]
val_data = data[split:]

if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:
    raise ValueError(
        "Dataset is too small for the current BLOCK_SIZE. "
        "Add more text or reduce BLOCK_SIZE."
    )


def get_batch(split_name):
    source = train_data if split_name == "train" else val_data

    ix = torch.randint(
        len(source) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[i:i + BLOCK_SIZE] for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + BLOCK_SIZE + 1] for i in ix
    ])

    return x.to(DEVICE), y.to(DEVICE)


# -----------------------------
# INITIALIZE MODEL
# -----------------------------

model = Amu(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    heads=HEADS,
    layers=LAYERS,
    block_size=BLOCK_SIZE
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Vocabulary size:", vocab_size)
print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)


# -----------------------------
# TRAINING
# -----------------------------

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for name in ["train", "val"]:
        losses = []

        for _ in range(20):
            x, y = get_batch(name)
            _, loss = model(x, y)
            losses.append(loss.item())

        results[name] = float(np.mean(losses))

    model.train()
    return results


for step in range(MAX_STEPS):

    if step % 500 == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss()

        print(
            f"Step {step}: "
            f"Train loss {losses['train']:.4f}, "
            f"Validation loss {losses['val']:.4f}"
        )

    x, y = get_batch("train")

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    optimizer.step()

# -----------------------------
# SAVE MODEL
# -----------------------------

os.makedirs("checkpoints", exist_ok=True)

torch.save(
    {
        "model_state": model.state_dict(),
        "vocab": characters,
        "config": {
            "vocab_size": vocab_size,
            "embed_dim": EMBED_DIM,
            "heads": HEADS,
            "layers": LAYERS,
            "block_size": BLOCK_SIZE
        }
    },
    "checkpoints/amu.pt"
)

print("Training completed!")
print("Model saved successfully.")

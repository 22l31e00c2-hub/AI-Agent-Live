"""Train amu from scratch on data/input.txt.

Run:  python train.py

Env overrides (handy for a smoke test, or to reproduce a run):
      AMU_STEPS=50       python train.py
      AMU_LR=1e-3        python train.py
      AMU_SEED=7         python train.py
      AMU_EVAL_EVERY=250 python train.py
      AMU_PATIENCE=500   python train.py

The schedule is the one current LLM training recipes use at every scale: a
short linear warmup, then cosine decay to a tenth of the peak rate, AdamW with
beta2 = 0.95, and decoupled weight decay on the matrices only.

The corpus is small, so this script keeps the checkpoint with the best
validation loss and stops once it has not improved for AMU_PATIENCE steps. On
this data that happens early; adding text to data/input.txt is the only real
fix for it.
"""

import math
import os
import time

import numpy as np
import torch

from model import Amu

# -----------------------------
# CONFIGURATION
# -----------------------------

BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_STEPS = int(os.environ.get("AMU_STEPS", "1000"))
LEARNING_RATE = float(os.environ.get("AMU_LR", "6e-4"))
WARMUP_STEPS = int(os.environ.get("AMU_WARMUP", "100"))
MIN_LR_RATIO = 0.1
EVAL_EVERY = int(os.environ.get("AMU_EVAL_EVERY", "100"))
EVAL_BATCHES = 20
PATIENCE = int(os.environ.get("AMU_PATIENCE", "300"))
SEED = int(os.environ.get("AMU_SEED", "1337"))

EMBED_DIM = 128
HEADS = 4
KV_HEADS = 2
LAYERS = 4

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

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
    kv_heads=KV_HEADS,
    layers=LAYERS,
    block_size=BLOCK_SIZE,
).to(DEVICE)

# Weight decay applies to matrices, never to norm scales or the embeddings.
decayed, undecayed = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    (decayed if param.dim() >= 2 else undecayed).append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decayed, "weight_decay": WEIGHT_DECAY},
        {"params": undecayed, "weight_decay": 0.0},
    ],
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
)

parameter_count = sum(p.numel() for p in model.parameters())

print("Vocabulary size:", vocab_size)
print("Model parameters:", parameter_count)
print(
    "Corpus: %d characters (%d train / %d validation)"
    % (len(data), len(train_data), len(val_data))
)
print(
    "Schedule: %d steps, lr %.1e -> %.1e, warmup %d, patience %d"
    % (MAX_STEPS, LEARNING_RATE, LEARNING_RATE * MIN_LR_RATIO, WARMUP_STEPS, PATIENCE)
)


def learning_rate_at(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return LEARNING_RATE * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine)


def save_checkpoint(path, losses, step):
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": characters,
            "config": {
                "vocab_size": vocab_size,
                "embed_dim": EMBED_DIM,
                "heads": HEADS,
                "kv_heads": KV_HEADS,
                "layers": LAYERS,
                "block_size": BLOCK_SIZE,
            },
            "meta": {
                "step": step,
                "steps_run": step + 1,
                "parameters": parameter_count,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "val_bits_per_char": losses["val"] / math.log(2),
                "seconds": round(time.perf_counter() - started, 2),
                "device": DEVICE,
                "seed": SEED,
            },
        },
        path,
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

        for _ in range(EVAL_BATCHES):
            x, y = get_batch(name)
            _, loss = model(x, y)
            losses.append(loss.item())

        results[name] = float(np.mean(losses))

    model.train()
    return results


os.makedirs("checkpoints", exist_ok=True)

started = time.perf_counter()

best = {"val": float("inf"), "step": -1, "train": float("inf")}
best_step = -1
completed = 0

for step in range(MAX_STEPS):

    for group in optimizer.param_groups:
        group["lr"] = learning_rate_at(step)

    if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss()

        print(
            f"Step {step}: "
            f"Train loss {losses['train']:.4f}, "
            f"Validation loss {losses['val']:.4f}, "
            f"val bits/char {losses['val'] / math.log(2):.3f}"
        )

        if losses["val"] < best["val"]:
            best = {
                "val": losses["val"],
                "step": step,
                "train": losses["train"],
            }
            best_step = step
            save_checkpoint("checkpoints/amu.pt", losses, step)
            print("    saved checkpoints/amu.pt (best validation so far)")

        completed = step + 1

        if step - best_step >= PATIENCE:
            print(
                "Stopping at step %d: validation has not improved since step %d."
                % (step, best_step)
            )
            break

    x, y = get_batch("train")

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRAD_CLIP
    )

    optimizer.step()

elapsed = time.perf_counter() - started
final_losses = estimate_loss()

# `amu.pt` holds the best validation weights; keep the last state too, so the
# overfitting is inspectable instead of hidden.
save_checkpoint("checkpoints/amu-final.pt", final_losses, completed - 1)

print("Training completed in %.1f s (%.1f steps/s)" % (elapsed, completed / elapsed))
print(
    "Final state: step %d, train %.4f, val %.4f"
    % (completed - 1, final_losses["train"], final_losses["val"])
)
print(
    "Best state : step %d, train %.4f, val %.4f (%.3f bits/char)"
    % (
        best_step,
        best["train"],
        best["val"],
        best["val"] / math.log(2),
    )
)
print("Saved checkpoints/amu.pt (best) and checkpoints/amu-final.pt (last).")

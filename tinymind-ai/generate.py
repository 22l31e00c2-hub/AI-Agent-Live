"""Sample text from your trained amu.

Interactive:  python generate.py
One shot:     python generate.py "LAYER " 200
"""

import sys

import torch

from model import Amu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model

checkpoint = torch.load(
    "checkpoints/amu.pt",
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


if len(sys.argv) > 1:
    prompt = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
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

"""Sample text from your trained amu.

Interactive:  python generate.py
One shot:     python generate.py "LAYER " 200
Top-k:        python generate.py --top-k 20 "LAYER " 200
Last state:   python generate.py --final "LAYER " 200
Benchmark:    python generate.py --bench

`checkpoints/amu.pt` holds the best-validation weights. `amu-final.pt` is the
last state of the run, which on a small corpus has memorised the text instead
of generalising - useful to look at, not the model to quote.

This module is also imported by mobile/server.py, so nothing here runs at import
time except the imports themselves. Call `load_model()` before `sample()`.
"""

import os
import re
import sys

import torch

from model import Amu
from cadagent.registry import all_recipes
from cadagent.social import respond as social_reply

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep the sample readable: the tail of a character distribution is noise.
TOP_K = 40

# Use the rule-based agent for conversational prompts so the model can answer
# greetings and identity questions coherently while the network is still small.
USE_AGENT_FOR_CHAT = True

_state = {}


def load_model(checkpoint="checkpoints/amu.pt"):
    """Load (and cache) a checkpoint. Returns the model, or raises."""
    if _state.get("path") == checkpoint and _state.get("model") is not None:
        return _state["model"]

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            "No checkpoint at %s yet. Run: python train.py" % checkpoint
        )

    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    characters = payload["vocab"]

    model = Amu(**payload["config"]).to(DEVICE)
    model.load_state_dict(payload["model_state"])
    model.eval()

    _state.update(
        path=checkpoint,
        model=model,
        characters=characters,
        stoi={ch: i for i, ch in enumerate(characters)},
        itos={i: ch for i, ch in enumerate(characters)},
    )
    return model


def encode(text):
    load_model() if "stoi" not in _state else None
    return [_state["stoi"][ch] for ch in text if ch in _state["stoi"]]


def decode(tokens):
    load_model() if "itos" not in _state else None
    return "".join(_state["itos"][int(i)] for i in tokens)


def _social_reply(prompt):
    """Best-effort conversational reply from the rule-based agent."""
    try:
        reply = social_reply(prompt, all_recipes())
    except Exception:
        return None
    if not reply:
        return None
    # Trim anything after a recipe-ish colon prefix if the reply started
    # with the prompt repeated by the agent.
    text = reply.strip()
    lowered = text.lower()
    if lowered.startswith(prompt.lower()[:60]):
        # keep only the conversational part after the prompt echo
        cut = text[len(prompt):].lstrip(" :")
        if cut:
            text = cut
    if not text:
        return None
    return text


def _is_conversational(prompt):
    """True when the prompt looks like a greeting / identity / help request."""
    t = prompt.lower()
    return bool(
        re.search(r"\b(hi|hey|hello|yo|good morning|good afternoon|good evening|thanks|thank you)\b", t)
        or "who are you" in t
        or "what are you" in t
        or "what can you do" in t
        or "your name" in t
        or "your creator" in t
        or "who made you" in t
        or "who created you" in t
        or "are you" in t
    )


def sample(prompt, max_new_tokens=200, temperature=0.8, top_k=TOP_K):
    # If the prompt is conversational and the agent has a rule-based reply,
    # use that first. The neural model is still too small to hold a coherent
    # conversation, so falling back here is the current intended behaviour.
    if USE_AGENT_FOR_CHAT and _is_conversational(prompt):
        reply = _social_reply(prompt)
        if reply:
            return reply

    model = load_model()
    tokens = encode(prompt)

    if not tokens:
        return None

    context = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    output = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    return decode(output[0].tolist())


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def _benchmark(prompt, limit, top_k):
    import time

    model = load_model()
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

    started = time.perf_counter()
    output = model.generate(
        context, max_new_tokens=limit, temperature=0.8, top_k=top_k
    )
    elapsed = max(time.perf_counter() - started, 1e-9)

    print("amu bench")
    print("params     : %d" % sum(p.numel() for p in model.parameters()))
    print("block size : %d" % model.block_size)
    print(
        "speed      : %d characters in %.2f s -> %.0f characters/s"
        % (limit, elapsed, limit / elapsed)
    )
    print("sample     : %r" % decode(output[0].tolist()))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    checkpoint = (
        "checkpoints/amu-final.pt" if "--final" in argv else "checkpoints/amu.pt"
    )
    flags = [value for value in argv if value != "--final"]

    top_k = TOP_K
    if "--top-k" in flags:
        index = flags.index("--top-k")
        if index + 1 >= len(flags):
            raise SystemExit("--top-k needs a number, for example: --top-k 40")
        top_k = int(flags[index + 1])
        del flags[index:index + 2]

    load_model(checkpoint)

    if flags and flags[0] in ("--bench", "-b"):
        prompt = flags[1] if len(flags) > 1 else "LAYER "
        limit = int(flags[2]) if len(flags) > 2 else 200
        _benchmark(prompt, limit, top_k)
        return 0

    if flags:
        prompt = flags[0]
        limit = int(flags[1]) if len(flags) > 1 else 200
        generated = sample(prompt, max_new_tokens=limit, top_k=top_k)
        if generated is None:
            print("AI: Please enter characters from my vocabulary.")
        else:
            print("AI:", generated)
        return 0

    print("amu is ready!")
    print("Type exit to quit.")

    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break

        if prompt.lower() == "exit":
            break

        generated = sample(prompt, top_k=top_k)

        if generated is None:
            print("AI: Please enter characters from my vocabulary.")
            continue

        print("\nAI:", generated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

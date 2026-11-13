/**
 * amu in the browser: ONNX Runtime + a character tokenizer + sampling.
 *
 * The runtime is injected so the same file works in a page
 * (onnxruntime-web) and in node (onnxruntime-node) for testing.
 *
 *   import { AmuEngine } from "./amu.js";
 *   const engine = await AmuEngine.create({ ort, modelUrl: "./model/amu.onnx" });
 *   const out = await engine.generate({ prompt: "hi amu,", maxTokens: 200 });
 *
 * The exported graph is a full-sequence forward pass, so each step re-runs the
 * whole window. block_size is small (128) and the model is ~3M parameters, so
 * that is cheaper than shipping a key/value cache across the wasm boundary.
 */

export class AmuEngine {
  constructor({ session, vocab, config, backend, ort }) {
    this.session = session;
    this.vocab = vocab;
    this.config = config;
    this.backend = backend;
    this.ort = ort;

    this.blockSize = config.block_size;
    this.vocabSize = config.vocab_size;

    this.stoi = new Map();
    vocab.forEach((ch, index) => this.stoi.set(ch, index));
  }

  static async create({ ort, modelUrl, vocab, config, preferWebGPU = true, providers = null }) {
    if (!vocab) {
      vocab = await (await fetch(modelUrl.replace(/[^/]*$/, "vocab.json"))).json();
    }
    if (!config) {
      config = await (await fetch(modelUrl.replace(/[^/]*$/, "config.json"))).json();
    }

    // Try the fastest backend first and fall back down the list. A device can
    // advertise WebGPU and still fail to build a session, so every entry is a
    // real attempt rather than a capability guess.
    let candidates = providers;
    if (!candidates) {
      const webgpu = preferWebGPU && !!(ort.env && ort.env.webgpu);
      candidates = webgpu ? ["webgpu", "wasm"] : ["wasm"];
    }

    let session = null;
    let backend = null;
    let lastError = null;

    for (const provider of candidates) {
      try {
        session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: [provider],
          graphOptimizationLevel: "all",
        });
        backend = provider;
        break;
      } catch (error) {
        lastError = error;
      }
    }

    if (!session) {
      throw new Error(
        "could not create an inference session (tried " +
          candidates.join(", ") +
          "): " +
          (lastError ? lastError.message : "unknown error")
      );
    }

    return new AmuEngine({ session, vocab, config, backend, ort });
  }

  encode(text) {
    const ids = [];
    for (const ch of text) {
      const id = this.stoi.get(ch);
      if (id !== undefined) ids.push(id);
    }
    return ids;
  }

  decode(ids) {
    return ids.map((id) => this.vocab[id]).join("");
  }

  /**
   * Continue `prompt`. Streaming: `onToken(char, fullText)` per character.
   * temperature <= 0 means greedy (deterministic).
   */
  async generate({
    prompt = "",
    maxTokens = 200,
    temperature = 0.8,
    topK = 40,
    onToken = null,
    signal = null,
  } = {}) {
    let ids = this.encode(prompt);
    if (ids.length === 0) {
      return { text: "", ids: [], seconds: 0, charsPerSecond: 0, reason: "no in-vocabulary characters" };
    }

    let text = prompt.slice();
    const startedAt = now();

    for (let step = 0; step < maxTokens; step++) {
      if (signal && signal.aborted) break;

      const window = ids.slice(-this.blockSize);
      const input = new BigInt64Array(window.map((id) => BigInt(id)));
      const tensor = new this.ort.Tensor("int64", input, [1, window.length]);
      const output = await this.session.run({ tokens: tensor });
      const logits = output.logits.data;
      const vocab = this.vocabSize;
      const offset = (window.length - 1) * vocab;

      const next = sampleFromLogits(logits, offset, vocab, temperature, topK);
      ids.push(next);

      const ch = this.vocab[next];
      text += ch;
      if (onToken) onToken(ch, text);
    }

    const seconds = (now() - startedAt) / 1000;
    const generated = ids.length - this.encode(prompt).length;

    return {
      text,
      ids,
      seconds,
      charsPerSecond: seconds > 0 ? generated / seconds : 0,
      generated,
      backend: this.backend,
    };
  }
}

function now() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/** Temperature + top-k sampling over one row of logits. */
export function sampleFromLogits(logits, offset, vocab, temperature, topK) {
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < vocab; i++) {
      if (logits[offset + i] > logits[offset + best]) best = i;
    }
    return best;
  }

  const scores = new Array(vocab);
  for (let i = 0; i < vocab; i++) scores[i] = logits[offset + i] / temperature;

  const k = Math.max(1, Math.min(topK || vocab, vocab));
  const ranked = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);

  const max = ranked[0].score;
  let total = 0;
  const weights = ranked.map((entry) => {
    const weight = Math.exp(entry.score - max);
    total += weight;
    return weight;
  });

  let pick = Math.random() * total;
  for (let i = 0; i < ranked.length; i++) {
    pick -= weights[i];
    if (pick <= 0) return ranked[i].index;
  }
  return ranked[ranked.length - 1].index;
}

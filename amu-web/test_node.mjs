/**
 * Run the browser engine in node so the wasm path is actually exercised.
 *
 *   npm install
 *   node test_node.mjs
 *
 * onnxruntime-node and onnxruntime-web share the same wasm kernels and the same
 * session API, so this proves the exported graph, the tokenizer and the sampling
 * loop work end to end. It does not exercise the page UI or WebGPU.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import * as ort from "onnxruntime-node";
import { AmuEngine } from "./amu.js";

const here = dirname(fileURLToPath(import.meta.url));

const vocab = JSON.parse(readFileSync(join(here, "model/vocab.json"), "utf8"));
const config = JSON.parse(readFileSync(join(here, "model/config.json"), "utf8"));

const engine = await AmuEngine.create({
  ort,
  modelUrl: join(here, "model/amu.onnx"),
  vocab,
  config,
  providers: ["cpu"],
});

console.log("backend     :", engine.backend);
console.log("vocab       :", engine.vocabSize, "characters");
console.log("block size  :", engine.blockSize);
console.log("parameters  :", config.parameters.toLocaleString("en-US"));

// 1) tokenizer round trip
const roundTrip = engine.decode(engine.encode("hi amu, create layers"));
if (roundTrip !== "hi amu, create layers") {
  throw new Error("tokenizer round trip failed: " + JSON.stringify(roundTrip));
}
console.log("tokenizer   : round trip ok");

// 2) greedy generation is deterministic
const greedyArgs = { prompt: "A layer groups", maxTokens: 40, temperature: 0 };
const a = await engine.generate(greedyArgs);
const b = await engine.generate(greedyArgs);
if (a.text !== b.text) {
  throw new Error("greedy generation is not deterministic");
}
console.log(`greedy      : deterministic, ${a.generated} chars in ${a.seconds.toFixed(3)}s`);
console.log("greedy text : %j", a.text);

// 3) sampling runs, streams, and respects the length limit
let streamed = 0;
const sampled = await engine.generate({
  prompt: "hi amu,",
  maxTokens: 30,
  temperature: 0.9,
  topK: 20,
  onToken: () => { streamed += 1; },
});
if (sampled.generated !== 30) {
  throw new Error("expected 30 generated characters, got " + sampled.generated);
}
if (streamed !== 30) {
  throw new Error("streaming callback fired " + streamed + " times, expected 30");
}
console.log(
  `sampling    : ${sampled.generated} chars, ${streamed} stream callbacks, ` +
    `${sampled.charsPerSecond.toFixed(1)} chars/s`
);
console.log("sampled text: %j", sampled.text);

// 4) a long run crosses block_size (128) and keeps producing
const long = await engine.generate({ prompt: "The command", maxTokens: 200, temperature: 0.8, topK: 40 });
if (long.generated !== 200) {
  throw new Error("long run produced " + long.generated + " characters");
}
console.log(
  `long run    : ${long.generated} chars across the ${engine.blockSize}-token window, ` +
    `${long.charsPerSecond.toFixed(1)} chars/s`
);

console.log("\nall node checks passed");

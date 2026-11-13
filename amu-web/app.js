/**
 * Chat controller for amu-web.
 *
 * The page is a thin shell: the model, the tokenizer and the sampling loop all
 * live in amu.js, which the node test drives too.
 */

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
import { AmuEngine } from "./amu.js";

const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
ort.env.wasm.wasmPaths = ORT_BASE;
// One thread: no SharedArrayBuffer, so no COOP/COEP headers and the folder can
// be served by anything, including a phone running Termux.
ort.env.wasm.numThreads = 1;

const $ = (id) => document.getElementById(id);

const header = $("header");
const scroller = $("scroller");
const thread = $("thread");
const empty = $("empty");
const form = $("composer");
const promptBox = $("prompt");
const sendButton = $("send");
const statusEl = $("status");
const statusText = $("status-text");
const modelBadge = $("model-badge");
const settings = $("settings");

let engine = null;
let busy = false;
let controller = null;

// ?backend=webgpu forces the GPU path, ?backend=wasm forces the CPU one.
//
// wasm is the default because it measured ~98 chars/s in Chrome while WebGPU
// managed ~0.8 chars/s on the same machine: a 3M-parameter model dispatches
// hundreds of tiny kernels per character, and GPU per-op overhead dominates.
// WebGPU stays available behind the flag for larger models or faster GPUs.
const forcedBackend = new URLSearchParams(location.search).get("backend");

/* ---------------------------------------------------------------- state */

function setStatus(text, kind = "") {
  statusText.textContent = text;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

function nearBottom() {
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
}

function scrollToBottom(force = false) {
  if (force || nearBottom()) {
    scroller.scrollTop = scroller.scrollHeight;
  }
}

/* ------------------------------------------------------------ rendering */

function hideEmptyState() {
  if (empty && empty.parentNode) empty.remove();
}

function addUserMessage(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  scrollToBottom(true);
}

function addAmuMessage() {
  const wrap = document.createElement("div");
  wrap.className = "msg amu";

  const who = document.createElement("div");
  who.className = "who";
  who.textContent = "amu";

  const body = document.createElement("div");
  body.className = "body";
  const textNode = document.createTextNode("");
  body.appendChild(textNode);

  const caret = document.createElement("span");
  caret.className = "caret";
  body.appendChild(caret);

  const actions = document.createElement("div");
  actions.className = "actions";

  const copy = document.createElement("button");
  copy.className = "copy";
  copy.type = "button";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(textNode.data);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Copy failed";
    }
    setTimeout(() => (copy.textContent = "Copy"), 1400);
  });
  actions.appendChild(copy);

  const meta = document.createElement("div");
  meta.className = "meta";

  wrap.append(who, body, actions, meta);
  thread.appendChild(wrap);
  scrollToBottom(true);

  return { wrap, body, textNode, caret, actions, meta };
}

/* -------------------------------------------------------------- engine */

async function boot() {
  setStatus("loading model", "busy");

  try {
    engine = await AmuEngine.create({
      ort,
      modelUrl: "./model/amu.onnx",
      preferWebGPU: false,
      providers: forcedBackend ? [forcedBackend] : null,
    });
  } catch (error) {
    setStatus("model unavailable", "err");
    modelBadge.textContent = "offline";
    promptBox.placeholder = "Could not load the model — " + error.message;
    hideEmptyState();
    const wrap = addAmuMessage();
    wrap.caret.remove();
    wrap.textNode.data = "The engine did not start: " + error.message;
    wrap.wrap.classList.add("error");
    return;
  }

  const params = engine.config.parameters
    ? (engine.config.parameters / 1e6).toFixed(1) + "M"
    : "—";

  modelBadge.textContent = `${params} params · ${engine.blockSize} ctx`;
  setStatus(`${engine.backend} ready`, "ok");

  promptBox.disabled = false;
  sendButton.disabled = false;
  promptBox.focus();
}

/* ------------------------------------------------------------ chatting */

async function generate(prompt) {
  if (!engine || busy) return;

  busy = true;
  controller = new AbortController();

  hideEmptyState();
  addUserMessage(prompt);

  const ui = addAmuMessage();
  ui.caret.remove();
  ui.body.appendChild(ui.caret);

  sendButton.classList.add("stop");
  sendButton.textContent = "■";
  sendButton.setAttribute("aria-label", "Stop");
  sendButton.disabled = false;
  setStatus("generating", "busy");

  const maxTokens = Number($("length").value);
  const temperature = Number($("temperature").value);
  const topK = Number($("topk").value);

  try {
    const result = await engine.generate({
      prompt,
      maxTokens,
      temperature,
      topK,
      signal: controller.signal,
      onToken: (char) => {
        ui.textNode.data += char;
        scrollToBottom();
      },
    });

    ui.caret.remove();
    ui.actions.classList.add("show");

    if (ui.textNode.data.length === 0) {
      ui.textNode.data = "(nothing generated)";
      ui.wrap.classList.add("error");
    } else {
      ui.meta.textContent = `${result.generated} chars · ${result.seconds.toFixed(2)}s · ${Math.round(
        result.charsPerSecond
      )} chars/s · ${result.backend}`;
    }

    setStatus(`${engine.backend} ready`, "ok");
  } catch (error) {
    ui.caret.remove();
    ui.actions.classList.add("show");
    ui.wrap.classList.add("error");
    ui.textNode.data = "Generation failed: " + error.message;
    setStatus("error", "err");
  } finally {
    busy = false;
    controller = null;
    sendButton.classList.remove("stop");
    sendButton.textContent = "↑";
    sendButton.setAttribute("aria-label", "Send");
    sendButton.disabled = promptBox.value.trim().length === 0;
    promptBox.focus();
  }
}

function submit() {
  const text = promptBox.value.trim();
  if (!text || busy || !engine) return;

  promptBox.value = "";
  autoGrow();
  sendButton.disabled = true;
  generate(text);
}

/* --------------------------------------------------------------- events */

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) {
    if (controller) controller.abort();
    return;
  }
  submit();
});

promptBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (busy) return;
    submit();
  }
});

function autoGrow() {
  promptBox.style.height = "auto";
  promptBox.style.height = Math.min(promptBox.scrollHeight, window.innerHeight * 0.4) + "px";
}

promptBox.addEventListener("input", () => {
  autoGrow();
  sendButton.disabled = busy || promptBox.value.trim().length === 0;
});

for (const chip of document.querySelectorAll(".suggestion")) {
  chip.addEventListener("click", () => {
    promptBox.value = chip.dataset.text;
    autoGrow();
    sendButton.disabled = false;
    promptBox.focus();
  });
}

for (const id of ["length", "temperature", "topk"]) {
  const input = $(id);
  const label = $(id + "-value");
  const format = () => {
    label.textContent =
      id === "temperature" ? Number(input.value).toFixed(2) : input.value;
  };
  input.addEventListener("input", format);
  format();
}

// Close the settings panel when tapping elsewhere.
document.addEventListener("click", (event) => {
  if (settings.open && !settings.contains(event.target)) settings.open = false;
});

scroller.addEventListener("scroll", () => {
  header.classList.toggle("scrolled", scroller.scrollTop > 4);
});

autoGrow();
boot();

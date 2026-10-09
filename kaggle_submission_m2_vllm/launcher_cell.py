# [calamitychasm] START SERVING: lordhansolo's vLLM stack instead of SGLang Pennyroyal.
# Built by scripts/_build_m2_vllm_kernel.py; see experiments/stage7_m2_speed.md ("vLLM stack").
#
# The previous cell wrote lordhansolo's own setup step (vLLM nightly e975732 unpacked from image
# layers, the hash-pinned 32-file overlay, his watchdog and GPU shard prefetcher) to
# VLLM_SETUP_SCRIPT, patched only where this harness needs it: max-model-len, max-num-seqs,
# max_pixels for 640 px boards, no reasoning_effort=xhigh, cached_tokens always reported, a loud
# VLLM_OVERLAY_HASH_MISMATCH line before anything is unpacked. This cell runs it in the background
# and, like the SGLang launcher it replaces, releases the benchmark at SERVER_STARTUP_TIMEOUT even
# if the server is still loading (the harness retries HTTP for ARC3_HTTP_RETRY_INITIAL_SECONDS).
# Once the server answers, boot probes log PREFIX_CACHE_PROBE / IMAGE_TOKENS_PROBE /
# REASONING_ECHO_PROBE / TEMPLATE_PROBE and the line VLLM_SERVING active.
# Required notebook variables: WORKING_DIR, NOTEBOOK_START_TIME, SERVER_STARTUP_TIMEOUT,
# SERVED_MODEL_NAME, SERVED_MODEL_PORT, TRUE_SUBMISSION, VLLM_RUNTIME_DIR, VLLM_MODEL_DIR,
# VLLM_BUNDLE_DIR.
import base64, hashlib, json, os, re, shutil, struct, subprocess, sys, threading, time, zlib
import urllib.request
from pathlib import Path

VLLM_MAX_NUM_SEQS = 14                 # [build] max-num-seqs; equals ARC3_MAX_ACTIVE_STREAMS
VLLM_MAX_MODEL_LEN = 139264            # [build] the SGLang context length the harness was tuned on
VLLM_IMAGE_MAX_PIXELS = 409600         # [build] 640x640: a 64x64 board at MULTIMODAL_UPSCALE 10
VLLM_SETUP_SCRIPT = Path("/kaggle/arc3_vllm_setup.py")
VLLM_SETUP_LOG = Path(WORKING_DIR) / "vllm-setup.log"
VLLM_SERVER_LOG = Path(WORKING_DIR) / "vllm-openai-server.log"
VLLM_WATCHDOG_LOG = Path(WORKING_DIR) / "vllm-watchdog.log"
VLLM_METRICS_LOG = Path(WORKING_DIR) / "vllm_metrics.jsonl"
VLLM_SERVER_URL = f"http://127.0.0.1:{SERVED_MODEL_PORT}"
# sha256 of the tokenizer.json the incumbent (SGLang, Intel W4A16) pins for its FR-Spec map
M2_TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
VLLM_METRIC_NAMES = (
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc", "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total", "vllm:request_success_total",
)


def _first_dir(*candidates):
    for c in candidates:
        if c and Path(c).is_dir():
            return Path(c)
    return None


def _http_json(path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(VLLM_SERVER_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat(messages, *, max_tokens=1, thinking=False, timeout=300):
    """One deterministic chat completion, shaped like the harness's own requests."""
    return _http_json("/v1/chat/completions", {
        "model": SERVED_MODEL_NAME, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "seed": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": bool(thinking), "preserve_thinking": True},
    }, timeout=timeout)


_PALETTE = ((0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0), (170, 170, 170),
            (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37), (255, 255, 255),
            (17, 17, 17), (57, 204, 204), (1, 255, 112), (177, 13, 201), (61, 153, 112))


def board_png_data_url(cells=64, scale=10, seed=0):
    """A cells x cells board of 16 colours upscaled `scale`x, as the harness renders a frame
    (MULTIMODAL_UPSCALE=10 -> 640x640), PNG-encoded without PIL."""
    side = cells * scale
    rows = []
    for gy in range(cells):
        row = bytearray(b"\x00")
        for gx in range(cells):
            row += bytes(_PALETTE[(gx * 7 + gy * 3 + seed + (gx * gy) // 5) % 16]) * scale
        rows.extend([bytes(row)] * scale)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _ascii_grid(seed, rows=32, cols=64):
    x, out = seed * 2654435761 % (1 << 32) or 1, []
    for _ in range(rows):
        line = []
        for _ in range(cols):
            x = (1103515245 * x + 12345) % (1 << 31)
            line.append("0123456789abcdef"[(x >> 16) % 16])
        out.append("".join(line))
    return "\n".join(out)


def probe_image_tokens():
    text = [{"role": "user", "content": [{"type": "text", "text": "Current grid image:"}]}]
    image = [{"role": "user", "content": [{"type": "text", "text": "Current grid image:"},
                                          {"type": "image_url", "image_url": {"url": board_png_data_url()}}]}]
    n0 = _chat(text)["usage"]["prompt_tokens"]
    n1 = _chat(image)["usage"]["prompt_tokens"]
    print(f"IMAGE_TOKENS_PROBE {n1 - n0} (one 640x640 board; the SGLang incumbent charges ~402; "
          f"~66 would mean max_pixels downscaled it)", flush=True)
    return n1 - n0


def probe_prefix_cache():
    """Same long multi-turn prompt (image + ASCII grids + prior reasoning) at temperature 0:
    cold, then twice warm. A Mamba/GDN state restored from the prefix cache must give the
    same tokens as a cold prefill; a mismatch is the hazard that cost the old vLLM stack 47%."""
    nonce = os.urandom(8).hex()
    msgs = [{"role": "system", "content": f"Probe {nonce}. You are a coding agent solving a grid-based puzzle game."}]
    for turn in range(4):
        content = [{"type": "text", "text": f"Turn {turn}. Grid (hex colours):\n{_ascii_grid(turn)}"}]
        if turn == 0:
            content.append({"type": "image_url", "image_url": {"url": board_png_data_url(seed=1)}})
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": f"I pressed ACTION{turn % 5 + 1}.",
                     os.environ.get("ARC3_REASONING_HISTORY_KEY", "reasoning"):
                         f"The pattern in turn {turn} repeats every {turn + 3} cells; try moving right."})
    msgs.append({"role": "user", "content": "Describe the last grid in one sentence and name your next action."})
    texts, cached, prompt = [], [], 0
    for _ in range(3):
        r = _chat(msgs, max_tokens=64)
        m = r["choices"][0]["message"]
        texts.append((m.get("content") or "") + "\x00" + (m.get("reasoning") or m.get("reasoning_content") or ""))
        u = r.get("usage") or {}
        prompt = u.get("prompt_tokens", 0)
        cached.append(((u.get("prompt_tokens_details") or {}).get("cached_tokens")))
    identical = len(set(texts)) == 1
    print(f"PREFIX_CACHE_PROBE identical={identical} prompt_tokens={prompt} cached_tokens={cached} "
          f"under_load={_PROBES_UNDER_LOAD}", flush=True)
    if not identical:
        for i, t in enumerate(texts):
            print(f"  PREFIX_CACHE_PROBE output[{i}]: {t[:300]!r}", flush=True)
    return identical


def probe_reasoning_echo():
    """Does the served chat template render a prior turn's reasoning under each message key?
    The harness echoes reasoning under ARC3_REASONING_HISTORY_KEY; a dropped block is invisible
    except as prompt growth that falls short of the tokens generated."""
    found, rendered = {}, {}
    for key in ("reasoning", "reasoning_content"):
        marker = f"MARKER-{key.upper()}-7F3A"
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "A1", key: f"thinking {marker}"},
                {"role": "user", "content": "U2"}]
        ids = _http_json("/tokenize", {"model": SERVED_MODEL_NAME, "messages": msgs, "add_generation_prompt": True,
                                       "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}})["tokens"]
        text = _http_json("/detokenize", {"model": SERVED_MODEL_NAME, "tokens": ids})["prompt"]
        found[key], rendered[key] = marker in text, text
    harness_key = os.environ.get("ARC3_REASONING_HISTORY_KEY", "reasoning").split(",")[0].strip()
    (Path(WORKING_DIR) / "vllm_rendered_probe.txt").write_text(rendered.get(harness_key, ""), encoding="utf-8")
    print(f"REASONING_ECHO_PROBE harness_key={harness_key} rendered={found.get(harness_key)} all={found}", flush=True)
    print(f"CHAT_RENDER_PROBE sha256={hashlib.sha256(rendered.get(harness_key, '').encode()).hexdigest()[:16]} "
          f"(full text in vllm_rendered_probe.txt)", flush=True)
    return found.get(harness_key)


def probe_template_files():
    """Hash and copy the served model's tokenizer / chat template / generation defaults, so the
    dev box can diff them against the incumbent's model (prompts depend on both)."""
    md = _first_dir(VLLM_MODEL_DIR, VLLM_MODEL_DIR.replace("/pytorch/", "/PyTorch/"))
    if md is None:
        print(f"TEMPLATE_PROBE model dir not found: {VLLM_MODEL_DIR}", flush=True)
        return
    out = Path(WORKING_DIR) / "vllm_model_meta"
    out.mkdir(exist_ok=True)
    shas = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                 "preprocessor_config.json", "config.json"):
        p = md / name
        if p.is_file():
            shas[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            shutil.copy2(p, out / name)
    print(f"TEMPLATE_PROBE tokenizer_sha={shas.get('tokenizer.json')} "
          f"matches_m2_pin={shas.get('tokenizer.json') == M2_TOKENIZER_SHA} "
          f"chat_template_sha={shas.get('chat_template.jinja')}", flush=True)
    gen = md / "generation_config.json"
    if gen.is_file():
        print("TEMPLATE_PROBE generation_config", json.dumps(json.loads(gen.read_text())), flush=True)


def parse_prometheus(text, names=VLLM_METRIC_NAMES):
    """Sum each wanted metric over its label sets (vLLM labels by model name and engine)."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in names:
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def _monitor(interval=120.0):
    """Check runs only: /metrics snapshots to vllm_metrics.jsonl, and watchdog lines (server
    exits / restarts) echoed into the notebook log so the check-run gate sees them."""
    seen = 0
    while True:
        try:
            with urllib.request.urlopen(VLLM_SERVER_URL + "/metrics", timeout=10) as resp:
                snap = parse_prometheus(resp.read().decode("utf-8", "replace"))
            with VLLM_METRICS_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), **snap}) + "\n")
        except Exception as exc:  # noqa: BLE001 - a dead server is the watchdog's to report
            print(f"[vllm-metrics] scrape failed: {exc!r}", flush=True)
        try:
            lines = VLLM_WATCHDOG_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[seen:]:
                print(f"[vllm-watchdog] {line}", flush=True)
            seen = len(lines)
        except OSError:
            pass
        time.sleep(interval)


_PROBES_UNDER_LOAD = False   # True when games already run beside the probes (batch-dependent numerics)


def _after_ready(proc, ready_at):
    """Runs once the setup step exited (the server is ready when it exits 0). ready_at None: the
    benchmark was already released at the deadline, so games share the server with the probes."""
    global _PROBES_UNDER_LOAD
    _PROBES_UNDER_LOAD = ready_at is None
    rc = proc.wait()
    _pump.join(timeout=30)
    if rc != 0:
        print(f"VLLM_SETUP_FAILED rc={rc}", flush=True)
        return False
    if not TRUE_SUBMISSION:
        for probe in (probe_template_files, probe_reasoning_echo, probe_image_tokens, probe_prefix_cache):
            try:
                probe()
            except Exception as exc:  # noqa: BLE001 - a probe never stops the run
                print(f"{probe.__name__.upper()} error={exc!r}", flush=True)
        threading.Thread(target=_monitor, name="vllm-monitor", daemon=True).start()
    print(f"VLLM_SERVING active port={SERVED_MODEL_PORT} model={SERVED_MODEL_NAME} max_num_seqs={VLLM_MAX_NUM_SEQS} "
          f"max_model_len={VLLM_MAX_MODEL_LEN} max_pixels={VLLM_IMAGE_MAX_PIXELS} "
          f"ready_after={int((ready_at or time.time()) - NOTEBOOK_START_TIME)}s", flush=True)
    return True


# ---- preflight: inputs present, and the notebook's Python is the one the image runtime was built for ----
_runtime = _first_dir(VLLM_RUNTIME_DIR, "/kaggle/input/vllm-main-e975732-arc3")
_bundle = _first_dir(VLLM_BUNDLE_DIR, "/kaggle/input/taaf-kaggle-source")
if _runtime is None or _bundle is None:
    print(f"VLLM_INPUT_MISSING runtime={_runtime} bundle={_bundle}", flush=True)
    raise FileNotFoundError("lordhansolo's vLLM runtime dataset or source bundle is not mounted")
_manifest = json.loads((_runtime / "runtime-manifest.json").read_text(encoding="utf-8"))
_abi = re.search(r"python3\.(\d+)", str(_manifest.get("dist_packages", "")))
if _abi and int(_abi.group(1)) != sys.version_info.minor:
    print(f"VLLM_PYTHON_ABI_MISMATCH notebook=3.{sys.version_info.minor} runtime=3.{_abi.group(1)}", flush=True)
    raise RuntimeError("the vLLM image runtime targets another Python; unpin docker_image in kernel-metadata.json")
print(f"vLLM runtime {_manifest.get('vllm_version')} ({_runtime}); draft vocabulary from {_bundle}", flush=True)

# ---- launch lordhansolo's setup step detached; echo its output into this log ----
_env = dict(os.environ, TAAF_KAGGLE_WORKING_DIR=str(WORKING_DIR), TAAF_KAGGLE_BUNDLE_DIR=str(_bundle),
            TAAF_KAGGLE_SETUP_ENV=str(Path(WORKING_DIR) / "vllm_setup_env.json"), PYTHONUNBUFFERED="1")
setup_proc = subprocess.Popen([sys.executable, "-u", str(VLLM_SETUP_SCRIPT)], env=_env, cwd=str(WORKING_DIR),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


def _pump_lines():
    with VLLM_SETUP_LOG.open("w", encoding="utf-8") as fh:
        for line in setup_proc.stdout:
            fh.write(line)
            fh.flush()
            print("[vllm-setup] " + line, end="", flush=True)


_pump = threading.Thread(target=_pump_lines, name="vllm-setup-log", daemon=True)
_pump.start()
print(f"vLLM setup pid {setup_proc.pid} -> {VLLM_SETUP_LOG}; server log {VLLM_SERVER_LOG}", flush=True)

_t0, _last = time.time(), -30
while time.time() - NOTEBOOK_START_TIME < SERVER_STARTUP_TIMEOUT and setup_proc.poll() is None:
    _elapsed = int(time.time() - _t0)
    if _elapsed - _last >= 30:
        print(f"  {_elapsed}s ...", flush=True)
        _last = _elapsed
    time.sleep(3)

if setup_proc.poll() is None:
    print(f"\nDEADLINE at {int(time.time() - NOTEBOOK_START_TIME)}s from notebook start; "
          f"releasing the benchmark with the vLLM server still loading", flush=True)
    threading.Thread(target=_after_ready, args=(setup_proc, None), name="vllm-after-ready", daemon=True).start()
elif setup_proc.returncode != 0:
    _pump.join(timeout=30)
    if VLLM_SERVER_LOG.exists():
        print(VLLM_SERVER_LOG.read_text(encoding="utf-8", errors="replace")[-20000:], flush=True)
    print(f"VLLM_SETUP_FAILED rc={setup_proc.returncode}", flush=True)
    raise RuntimeError(f"vLLM setup failed (rc={setup_proc.returncode}); see the [vllm-setup] lines above")
else:
    print(f"\nREADY after {int(time.time() - _t0)}s -> {VLLM_SERVER_URL}/v1", flush=True)
    _after_ready(setup_proc, time.time())

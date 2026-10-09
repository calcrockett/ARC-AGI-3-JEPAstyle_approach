"""The incumbent harness on lordhansolo's vLLM stack (scripts/_build_m2_vllm_kernel.py).

Kaggle is unreachable from the build machine, so everything a check run would otherwise be the
first to find is pinned here: the build is deterministic and up to date, every harness cell is
the incumbent's, no SGLang launch is left, the vLLM command carries this harness's knobs, a wrong
overlay prints its marker and stops, the launcher's boot probes run end to end against a fake
server, and the speed report reads vLLM logs.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import math
import os
import struct
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _build_m2_vllm_kernel as b  # noqa: E402
import m2_speed_report as rep  # noqa: E402
from _check_notebook_cell import check_notebook  # noqa: E402

LORD_CLONE = Path("/home/user/tonghuikang/daniel-franzen-arc-agi-3/kaggle/lordhansolo/dataset-taaf-kaggle-source")


def cells(path: Path) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("vllm_build")
    return {name: b.build(name, root) for name in b.VARIANTS}


INCUMBENT_NB = b.INCUMBENT / "arc3-m2-level-memory.ipynb"


def setup_cell(path: Path) -> str:
    [s] = [c for c in code_cells(path) if c.startswith(f"%%writefile {b.SETUP_PATH}\n")]
    return s.split("\n", 1)[1]


def launcher_cell(path: Path) -> str:
    [s] = [c for c in code_cells(path) if c.startswith("# [calamitychasm] START SERVING")]
    return s


# ---------------------------------------------------------------- build ----

def test_build_is_deterministic_and_committed_copies_are_current(built, tmp_path):
    for name, path in built.items():
        again = b.build(name, tmp_path)
        assert again.read_bytes() == path.read_bytes()
        assert (again.parent / "kernel-metadata.json").read_bytes() == (path.parent / "kernel-metadata.json").read_bytes()
        committed = b.out_dir(name) / path.name
        assert committed.read_bytes() == path.read_bytes(), f"rerun scripts/_build_m2_vllm_kernel.py ({name})"
        assert (committed.parent / "kernel-metadata.json").read_bytes() == (path.parent / "kernel-metadata.json").read_bytes()


def test_cells_parse_and_pass_use_before_definition(built):
    for path in built.values():
        assert check_notebook(path) == []
        for src in code_cells(path):
            if src.startswith("%%writefile"):
                body = src.split("\n", 1)[1]
                if src.startswith(f"%%writefile {b.SETUP_PATH}"):
                    ast.parse(body)
            else:
                ast.parse(src.replace("\n!", "\n#!"))


def test_incumbent_knobs_are_read_not_retyped():
    knobs = b.incumbent_serving(cells(INCUMBENT_NB))
    assert knobs == {"max_model_len": (116 + 12 + 8) * 1024, "max_pixels": 640 * 640}
    assert knobs["max_pixels"] >= 409600


def resolved_incumbent_cells():
    """The incumbent as every derived kernel sees it: with the mount-layout input resolver applied (the
    committed incumbent v1 itself has none; see tests/test_m2_input_resolver.py)."""
    import _m2_input_resolver
    nb = json.loads(INCUMBENT_NB.read_text(encoding="utf-8"))
    _m2_input_resolver.apply_input_resolver(nb)
    return ["".join(c["source"]) for c in nb["cells"]]


def test_only_serving_cells_differ_from_the_incumbent(built):
    inc = resolved_incumbent_cells()
    for name, path in built.items():
        new = cells(path)
        assert new[0].startswith(f"## [calamitychasm] {b.kernel_slug(name)}")
        new = new[1:]
        launcher = next(i for i, s in enumerate(inc) if 'PREFIX = "/tmp/sgl-intel"' in s)
        # incumbent [..., md5, sglang] -> [..., md5', writefile, launcher]
        assert len(new) == len(inc) + 1
        same_before = new[:launcher - 1]
        same_after = new[launcher + 2:]
        changed = {i for i, (x, y) in enumerate(zip(inc[:launcher - 1], same_before)) if x != y}
        changed |= {launcher + 1 + i for i, (x, y) in enumerate(zip(inc[launcher + 1:], same_after)) if x != y}
        edited = {next(i for i, s in enumerate(inc) if "'ARC3_REASONING_HISTORY_KEY'" in s),
                  next(i for i, s in enumerate(inc) if "def precache(" in s),
                  next(i for i, s in enumerate(inc) if "demo_excluded_games = [] if TRUE_SUBMISSION" in s)}
        assert changed == edited
        # the harness patch, level memory install and the run cell are untouched
        for marker in ("%%writefile /kaggle/harness-changes.patch", "LEVEL_MEMORY.install(", "print('Starting benchmark...')"):
            assert [s for s in inc if marker in s] == [s for s in new if marker in s]


def test_no_sglang_launch_remains(built):
    for path in built.values():
        for src in code_cells(path):
            if src.startswith("%%writefile /kaggle/harness-changes.patch"):
                continue
            code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
            assert "xhigh" not in src, path.name
            for gone in ("sglang serve", "sgl-intel", "pennyroyal-v253", "SGLANG", "dfranzen/intel", "albucino",
                         "DRAFT_MODEL_DIR", "WHEELHOUSE_DIR", "--model-path", "--mem-fraction-static", "SERVED_MODEL_PORT = 8001"):
                assert gone not in code, (path.name, gone)


def test_env_cell_and_flags(built):
    for name, path in built.items():
        n = b.VARIANTS[name]
        env = next(s for s in code_cells(path) if "'ARC3_REASONING_HISTORY_KEY'" in s)
        assert "'ARC3_REASONING_HISTORY_KEY': 'reasoning'," in env
        assert f"'ARC3_MAX_ACTIVE_STREAMS': {n}," in env
        assert "SERVED_MODEL_PORT = 1234" in env and "SERVED_MODEL_NAME = 'flashnext'" in env
        # the paths are resolved against whichever /kaggle/input layout this session mounted
        assert ("VLLM_RUNTIME_DIR  = resolve_input('vllm_runtime', "
                "'/kaggle/input/datasets/lordhansolo/vllm-main-e975732-arc3')") in env
        assert ("VLLM_MODEL_DIR    = resolve_input('vllm_model', "
                "'/kaggle/input/models/lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/pytorch/hf-mixed-mtp-nvfp4/1')") in env
        assert ("VLLM_BUNDLE_DIR   = resolve_input('vllm_bundle', "
                "'/kaggle/input/datasets/lordhansolo/taaf-kaggle-source')") in env
        assert ("ORIG_BUNDLE_DIR   = resolve_input('orig_bundle', "
                "'/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy')") in env
        # the harness's own sampling stays what the incumbent sends per request
        for k in ("'LOCAL_ANALYZER_TEMPERATURE': '0.7'", "'LOCAL_ANALYZER_TOP_P': '0.95'", "'LOCAL_ANALYZER_TOP_K': '20'"):
            assert k in env
        custom = next(s for s in code_cells(path) if "demo_excluded_games" in s)
        assert "demo_excluded_games = []   #" in custom
        launcher = launcher_cell(path)
        assert f"VLLM_MAX_NUM_SEQS = {n} " in launcher and "VLLM_MAX_MODEL_LEN = 139264 " in launcher
        assert "VLLM_IMAGE_MAX_PIXELS = 409600 " in launcher
        assert f'VLLM_SETUP_SCRIPT = Path({b.SETUP_PATH!r})' in launcher
        setup = setup_cell(path)
        assert f"\nVLLM_MAX_NUM_SEQS = {n}  #" in setup
        assert "\nVLLM_MAX_MODEL_LEN = 139264  #" in setup and "\nVLLM_IMAGE_MAX_PIXELS = 409600  #" in setup
        assert "'{\"preserve_thinking\": true}'," in setup and "\"reasoning_effort\"" not in setup
        assert "    cmd += ['--enable-prompt-tokens-details']  #" in setup
        assert "'--generation-config',\n        'vllm'," in setup
        assert "VLLM_SETUP_DONE" in setup and "LOCAL_ANALYZER_TEMPERATURE" not in setup


def test_metadata_sources(built):
    for name, path in built.items():
        meta = json.loads((path.parent / "kernel-metadata.json").read_text())
        assert meta["id"] == f"calamitychasm/arc3-m2-vllm-{name}" and meta["code_file"] == path.name
        assert meta["dataset_sources"] == ["lordhansolo/vllm-main-e975732-arc3", "lordhansolo/taaf-kaggle-source",
                                           "dfranzen/taaf-kaggle-source-bundle-copy"]
        assert meta["model_sources"] == ["lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1"]
        assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
        assert meta["machine_shape"] == "NvidiaRtxPro6000" and meta["enable_gpu"] is True
        assert meta["enable_internet"] is False and meta["is_private"] is True
        inc = json.loads((b.INCUMBENT / "kernel-metadata.json").read_text())
        assert meta["docker_image_pinning_type"] == "original" and meta["docker_image"] == inc["docker_image"]
        assert b.out_dir(name).name == "notebook" and b.out_dir(name).parent.name == f"kaggle_submission_m2_vllm_{name}"


def test_vendored_setup_is_verbatim():
    text = b.SETUP_SRC.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode()).hexdigest() == b.SETUP_BODY_SHA256
    prov = json.loads((b.SETUP_SRC.parent / "PROVENANCE.json").read_text())
    assert prov["body_sha256"] == b.SETUP_BODY_SHA256
    if (LORD_CLONE / "setup_commands.json").exists():
        raw = (LORD_CLONE / "setup_commands.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == prov["source_file_sha256"]
        [cmd] = json.loads(raw)
        assert cmd == "\"$PYTHON\" - <<'PYSETUP'\n" + text + "PYSETUP"


# ------------------------------------------- the patched setup step itself ----

def _load_setup_defs(src: str, env: dict) -> dict:
    """Imports, constant assignments and defs of the setup step, without its top-level actions."""
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign))]
    ns: dict = {"__name__": "setup_under_test"}
    old = dict(os.environ)
    os.environ.update(env)
    try:
        exec(compile(ast.Module(body=keep, type_ignores=[]), "arc3_vllm_setup.py", "exec"), ns)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return ns


@pytest.mark.parametrize("name", list(b.VARIANTS))
def test_vllm_command_carries_this_harness_knobs(built, name, tmp_path):
    (tmp_path / "bundle" / "src" / "ARC3-Inference" / "configs").mkdir(parents=True)
    ns = _load_setup_defs(setup_cell(built[name]), {"TAAF_KAGGLE_WORKING_DIR": str(tmp_path),
                                                     "TAAF_KAGGLE_BUNDLE_DIR": str(tmp_path / "bundle")})
    cmd = ns["build_vllm_server_command"]()
    n = b.VARIANTS[name]

    def arg(flag):
        return cmd[cmd.index(flag) + 1]
    assert arg("--max-num-seqs") == str(n) and arg("--max-model-len") == "139264"
    assert arg("--max-cudagraph-capture-size") == str(math.ceil(n * 4 / 8) * 8)
    assert json.loads(arg("--mm-processor-kwargs")) == {"max_pixels": 409600}
    assert json.loads(arg("--default-chat-template-kwargs")) == {"preserve_thinking": True}
    assert json.loads(arg("--limit-mm-per-prompt")) == {"image": 64, "video": 0}
    i = cmd.index("--served-model-name")
    assert cmd[i + 1:i + 3] == ["flashnext", b.LORD_MODEL_NAME]
    assert arg("--generation-config") == "vllm" and arg("--port") == "1234"
    assert arg("--reasoning-parser") == "qwen3" and arg("--tool-call-parser") == "qwen3_coder"
    assert "--enable-auto-tool-choice" in cmd and "--enable-prefix-caching" in cmd
    assert json.loads(arg("--speculative-config"))["num_speculative_tokens"] == 3
    assert arg("--kv-cache-dtype") == "fp8_e4m3" and arg("--gpu-memory-utilization") == "0.98"
    # cache tracing is opt-in, so a check run is not slowed by per-lookup tracing
    env: dict = {}
    os.environ.pop("ARC3_VLLM_CACHE_DIAGNOSTICS", None)
    os.environ["TAAF_RUN_AS_SUBMISSION"], saved = "0", os.environ.get("TAAF_RUN_AS_SUBMISSION")
    try:
        ns["configure_cache_diagnostics"]({"dist_packages": "x"}, env, cmd)
    finally:
        if saved is None:
            os.environ.pop("TAAF_RUN_AS_SUBMISSION")
        else:
            os.environ["TAAF_RUN_AS_SUBMISSION"] = saved
    assert env["ARC3_CACHE_DIAGNOSTICS"] == "0" and "--kv-cache-metrics" not in cmd


def _gate_ns(built, tmp_path):
    ns = _load_setup_defs(setup_cell(built["s12"]), {"TAAF_KAGGLE_WORKING_DIR": str(tmp_path),
                                                     "TAAF_KAGGLE_BUNDLE_DIR": str(tmp_path)})
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ns["RUNTIME_DATASET"] = runtime
    return ns, runtime


def test_overlay_hash_mismatch_prints_marker_and_stops(built, tmp_path, capsys):
    ns, runtime = _gate_ns(built, tmp_path)
    (runtime / "arc3_vllm_main_e975732_arc3_overlay.tar.blob").write_bytes(b"not the pinned overlay")
    found = hashlib.sha256(b"not the pinned overlay").hexdigest()
    manifest = {"patch": dict(ns["FINE_PREFIX_CACHE_PATCH_IDENTITY"])}
    with pytest.raises(SystemExit) as exc:
        ns["arc3_overlay_hash_gate"](manifest)
    assert exc.value.code == 86
    assert f"VLLM_OVERLAY_HASH_MISMATCH {found} expected=e3a6fe0d" in capsys.readouterr().out
    (runtime / "arc3_vllm_main_e975732_arc3_overlay.tar.blob").unlink()
    with pytest.raises(SystemExit):
        ns["arc3_overlay_hash_gate"](manifest)
    assert "VLLM_OVERLAY_HASH_MISMATCH missing" in capsys.readouterr().out


def test_overlay_hash_gate_passes_a_matching_overlay_and_checks_the_manifest(built, tmp_path, capsys):
    ns, runtime = _gate_ns(built, tmp_path)
    blob = b"pinned overlay bytes"
    (runtime / "arc3_vllm_main_e975732_arc3_overlay.tar.blob").write_bytes(blob)
    ns["FINE_PREFIX_CACHE_PATCH_IDENTITY"]["overlay_sha256"] = hashlib.sha256(blob).hexdigest()
    overlays = [{"target": t, **h} for t, h in ns["FINE_PREFIX_CACHE_REQUIRED_OVERLAYS"].items()]
    manifest = {**ns["FINE_PREFIX_CACHE_RUNTIME_IDENTITY"],
                "patch": {**ns["FINE_PREFIX_CACHE_PATCH_IDENTITY"], "overlay_files": overlays}}
    ns["arc3_overlay_hash_gate"](manifest)
    assert f"VLLM_OVERLAY_HASH_OK {hashlib.sha256(blob).hexdigest()}" in capsys.readouterr().out
    manifest["patch"]["overlay_files"] = overlays[1:]
    with pytest.raises(SystemExit):
        ns["arc3_overlay_hash_gate"](manifest)
    assert "VLLM_OVERLAY_HASH_MISMATCH manifest-identity" in capsys.readouterr().out


# --------------------------------------------- launcher against a fake server ----

class FakeVLLM(BaseHTTPRequestHandler):
    seen_reasoning_keys = ("reasoning", "reasoning_content")

    def log_message(self, *a):  # noqa: D401 - keep pytest output clean
        pass

    def _send(self, obj, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/metrics":
            self._send('# HELP x\nvllm:num_preemptions_total{model_name="flashnext"} 3.0\n'
                       'vllm:prefix_cache_queries_total{model_name="flashnext"} 100.0\n'
                       'vllm:prefix_cache_hits_total{model_name="flashnext"} 90.0\n', "text/plain")
        else:
            self._send({"data": [{"id": "flashnext"}]})

    def do_POST(self):  # noqa: N802
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            images, text = 0, []
            for m in req["messages"]:
                content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
                for part in content:
                    if part["type"] == "image_url":
                        png = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                        w, h = struct.unpack(">II", png[16:24])
                        images += 1 if (w, h) == (640, 640) else 0
                    else:
                        text.append(part["text"])
            prompt = len("".join(text)) // 4 + 402 * images
            self._send({"choices": [{"message": {"content": "Move right.", "reasoning": "r"}}],
                        "usage": {"prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": 0}}})
        elif self.path == "/tokenize":
            rendered = "".join(f"<{m['role']}>{m.get('reasoning', '')}{m['content']}" for m in req["messages"])
            self._send({"tokens": [ord(c) for c in rendered]})
        elif self.path == "/detokenize":
            self._send({"prompt": "".join(chr(t) for t in req["tokens"])})


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def _launcher_ns(tmp_path, port, setup_body, timeout=30):
    runtime, model, bundle = tmp_path / "runtime", tmp_path / "model", tmp_path / "bundle"
    for d in (runtime, model, bundle):
        d.mkdir()
    (runtime / "runtime-manifest.json").write_text(json.dumps({
        "dist_packages": f"usr/local/lib/python3.{sys.version_info.minor}/dist-packages", "vllm_version": "test"}))
    (model / "tokenizer.json").write_text("{}")
    (model / "generation_config.json").write_text('{"temperature": 0.6}')
    setup = tmp_path / "setup.py"
    setup.write_text(setup_body)
    src = b.launcher_source(12, 139264, 409600).replace(repr(b.SETUP_PATH), repr(str(setup)))
    ns = {"__name__": "__main__", "WORKING_DIR": tmp_path, "NOTEBOOK_START_TIME": time.time(),
          "SERVER_STARTUP_TIMEOUT": timeout, "SERVED_MODEL_NAME": "flashnext", "SERVED_MODEL_PORT": port,
          "TRUE_SUBMISSION": False, "VLLM_RUNTIME_DIR": str(runtime), "VLLM_MODEL_DIR": str(model),
          "VLLM_BUNDLE_DIR": str(bundle)}
    return src, ns


def test_launcher_ready_path_runs_probes_and_marks_serving(tmp_path, fake_server, capsys, monkeypatch):
    monkeypatch.setenv("ARC3_REASONING_HISTORY_KEY", "reasoning")
    src, ns = _launcher_ns(tmp_path, fake_server, "print('VLLM_OVERLAY_HASH_OK abc')\nprint('VLLM_SETUP_DONE')\n")
    exec(compile(src, "launcher_cell", "exec"), ns)
    out = capsys.readouterr().out
    assert "[vllm-setup] VLLM_OVERLAY_HASH_OK abc" in out and "READY after" in out
    assert "IMAGE_TOKENS_PROBE 402 " in out and "MULTI_IMAGE_PROBE images=6 ok=True tokens=2412" in out
    assert "PREFIX_CACHE_PROBE identical=True" in out and "under_load=False" in out
    assert "REASONING_ECHO_PROBE harness_key=reasoning rendered=True" in out
    assert "TEMPLATE_PROBE tokenizer_sha=" in out and "matches_m2_pin=False" in out
    assert "VLLM_SERVING active port=" in out and "max_num_seqs=12" in out
    assert (tmp_path / "vllm-setup.log").read_text().startswith("VLLM_OVERLAY_HASH_OK")
    assert (tmp_path / "vllm_model_meta" / "generation_config.json").exists()
    for _ in range(50):   # the monitor thread's first /metrics snapshot
        if (tmp_path / "vllm_metrics.jsonl").exists():
            break
        time.sleep(0.1)
    snap = json.loads((tmp_path / "vllm_metrics.jsonl").read_text().splitlines()[0])
    assert snap["vllm:num_preemptions_total"] == 3.0 and snap["vllm:prefix_cache_hits_total"] == 90.0


def test_launcher_fails_fast_on_overlay_mismatch(tmp_path, fake_server, capsys):
    src, ns = _launcher_ns(tmp_path, fake_server,
                           "import sys\nprint('VLLM_OVERLAY_HASH_MISMATCH deadbeef expected=e3a6')\nsys.exit(86)\n")
    with pytest.raises(RuntimeError, match="rc=86"):
        exec(compile(src, "launcher_cell", "exec"), ns)
    out = capsys.readouterr().out
    assert "[vllm-setup] VLLM_OVERLAY_HASH_MISMATCH deadbeef" in out and "VLLM_SETUP_FAILED rc=86" in out
    assert "VLLM_SERVING active" not in out


def test_launcher_releases_the_benchmark_at_the_deadline(tmp_path, fake_server, capsys):
    src, ns = _launcher_ns(tmp_path, fake_server, "import time\ntime.sleep(6)\nprint('VLLM_SETUP_DONE')\n", timeout=0.5)
    exec(compile(src, "launcher_cell", "exec"), ns)
    assert "DEADLINE at" in capsys.readouterr().out
    [t] = [t for t in threading.enumerate() if t.name == "vllm-after-ready"]
    t.join(timeout=60)
    out = capsys.readouterr().out
    assert "PREFIX_CACHE_PROBE identical=True" in out and "under_load=True" in out
    assert "VLLM_SERVING active" in out


def test_launcher_refuses_a_python_abi_mismatch(tmp_path, fake_server, capsys):
    src, ns = _launcher_ns(tmp_path, fake_server, "print('never runs')\n")
    (tmp_path / "runtime" / "runtime-manifest.json").write_text(json.dumps({"dist_packages": "usr/lib/python3.1/x"}))
    with pytest.raises(RuntimeError, match="another Python"):
        exec(compile(src, "launcher_cell", "exec"), ns)
    assert "VLLM_PYTHON_ABI_MISMATCH" in capsys.readouterr().out


def test_board_png_is_a_640px_rgb_image():
    ns: dict = {}
    tree = ast.parse(b.LAUNCHER_SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n.name == "board_png_data_url")
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_PALETTE")]
    exec(compile(ast.Module(body=keep, type_ignores=[]), "launcher", "exec"), ns)
    png = base64.b64decode(ns["board_png_data_url"]().split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">IIBB", png[16:26]) == (640, 640, 8, 2)
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(io.BytesIO(png))
    assert im.size == (640, 640) and im.mode == "RGB" and len(im.getcolors(1 << 16)) > 4


# ------------------------------------------------------ speed report ----

SYNTH_SERVER_LOG = """\
INFO 10-10 01:00:00 [gpu_worker.py:1] GPU KV cache size: 1,234,560 tokens
INFO 10-10 01:00:10 [loggers.py:1] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
""" + "".join(
    f"INFO 10-10 01:{i:02d}:20 [loggers.py:1] Engine 000: Avg prompt throughput: 900.0 tokens/s, "
    f"Avg generation throughput: {600 + 10 * i}.0 tokens/s, Running: {12 if i % 2 else 11} reqs, Waiting: {i % 3} reqs, "
    f"GPU KV cache usage: {50 + i}.5%, Prefix cache hit rate: {80 + i / 10:.1f}%\n"
    f"INFO 10-10 01:{i:02d}:20 [metrics.py:1] SpecDecoding metrics: Mean acceptance length: 2.{60 + i}, "
    f"Accepted throughput: 1.0 tokens/s, Drafted throughput: 2.0 tokens/s\n" for i in range(20)) + """\
==== watchdog restart 1 at 2026-10-10T02:00:00 ====
ERROR torch.AcceleratorError: CUDA error: an illegal memory access was encountered
"""


def test_speed_report_parses_a_vllm_run(tmp_path):
    r = rep.parse_vllm_server_log(SYNTH_SERVER_LOG)
    assert r["kv_pool"] == "1234560" and r["restarts"] == 1 and r["cuda_faults"] == 2
    assert 11 <= r["running_med"] <= 12 and set(r["by_running"]) == {11, 12}
    assert r["decode_med"] == 695 and r["kv_peak"] == pytest.approx(0.695)
    assert r["accept_med"] == pytest.approx(2.695, abs=0.01) and r["waiting_max"] == 2
    assert r["prefix_hit_last"] == pytest.approx(81.9)
    met = rep.parse_vllm_metrics([
        json.dumps({"ts": 1, "vllm:num_preemptions_total": 1.0}),
        "not json",
        json.dumps({"ts": 2, "vllm:num_preemptions_total": 7.0, "vllm:prefix_cache_queries_total": 200.0,
                    "vllm:prefix_cache_hits_total": 180.0, "vllm:spec_decode_num_drafts_total": 100.0,
                    "vllm:spec_decode_num_accepted_tokens_total": 170.0, "vllm:generation_tokens_total": 5e6}),
    ])
    assert met == {"preempts": 7, "prefix_hit": 90.0, "accept_len": 2.7, "generated": 5000000}

    d = tmp_path / "run"
    d.mkdir()
    (d / "summary.txt").write_text("generated tokens/sec: 701.5\nmean score: 12.0\nlevels=3/7 levels=2/5\n")
    (d / "vllm-openai-server.log").write_text(SYNTH_SERVER_LOG)
    (d / "vllm-watchdog.log").write_text("2026 restarting vLLM server in 5s (restart 1 of 5)\n")
    (d / "vllm_metrics.jsonl").write_text(json.dumps({"vllm:num_preemptions_total": 4.0,
                                                      "vllm:prefix_cache_queries_total": 10.0,
                                                      "vllm:prefix_cache_hits_total": 9.0}) + "\n")
    (d / "arc3-m2-vllm-s12.log").write_text(json.dumps([
        {"data": "VLLM_OVERLAY_HASH_OK e3a6\nTEMPLATE_PROBE tokenizer_sha=abc matches_m2_pin=True chat_template_sha=x\n"},
        {"data": "REASONING_ECHO_PROBE harness_key=reasoning rendered=True all={}\nIMAGE_TOKENS_PROBE 402 (one)\n"},
        {"data": "MULTI_IMAGE_PROBE images=6 ok=True tokens=2412\n"},
        {"data": "PREFIX_CACHE_PROBE identical=True prompt_tokens=9000 cached_tokens=[0, 8960, 8960]\nVLLM_SERVING active\n"}]))
    out = rep.analyse("vllm-s12", d)
    assert out["gen_tok_s"] == "701.5" and out["levels"] == 5
    assert out["preempts"] == 4 and out["prefix_hit"] == 90.0 and out["restarts"] == 2
    assert out["kv_pool"] == "1234560" and out["retracts"] == 0 and out["tracebacks"] == 0
    assert out["probes"] == {"PREFIX_CACHE_PROBE": "True", "IMAGE_TOKENS_PROBE": "402", "MULTI_IMAGE_PROBE": "True",
                             "REASONING_ECHO_PROBE": "True",
                             "TEMPLATE_PROBE": "True", "VLLM_OVERLAY_HASH": "OK", "VLLM_SERVING": "active"}
    assert rep.slug("vllm-s14") == "arc3-m2-vllm-s14" and rep.slug("base") == "arc3-m2-speed-base"


def test_speed_report_sglang_parsing_unchanged(tmp_path):
    d = tmp_path / "base"
    d.mkdir()
    (d / "summary.txt").write_text("generated tokens/sec: 577.8\n")
    (d / "serve.log").write_text("".join(
        f"Decode batch, #running-req: 10, full token usage: 0.9{i}, mamba usage: 0.5, accept len: 2.6, "
        f"gen throughput (token/s): {770 + i}.0\n" for i in range(6)) + "max_total_num_tokens=1010000\n")
    out = rep.analyse("base", d)
    assert out["decode_med"] == 772 and out["by_running"] == {10: 772} and out["kv_pool"] == "1010000"
    assert "probes" not in out and "preempts" not in out

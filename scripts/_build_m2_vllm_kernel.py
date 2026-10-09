"""Build arc3-m2-vllm-s12 / -s14: the incumbent harness on lordhansolo's vLLM serving stack.

Source: the incumbent notebook (kaggle_submission_m2_level_memory/notebook, built by
scripts/_build_m2_level_memory_kernel.py -- milestone-2 harness + solved-level memory). Only the
serving changes; every harness cell, the level-memory install and the run cell stay byte-identical
except where an anchor below says otherwise:

  cell "1. Environment"  model/runtime paths -> lordhansolo's datasets + model; port 1234;
                         ARC3_REASONING_HISTORY_KEY=reasoning (vLLM's field); ARC3_MAX_ACTIVE_STREAMS=N
  cell "2. Precaching"   no Pennyroyal wheelhouse / dfranzen model to precache (his watchdog runs a
                         GPU shard prefetcher instead)
  cell "5. Start serving" SGLang launcher -> (a) %%writefile of lordhansolo's own setup step
                         (kaggle_submission_m2_vllm/vendor/lordhansolo_vllm_setup.py, vendored verbatim,
                         patched here by exact anchors: SETUP_PATCHES) and (b) a launcher cell
                         (kaggle_submission_m2_vllm/launcher_cell.py) that runs it detached, keeps the
                         12-minute release, and logs boot probes
  cell "7. Customization" check run plays all 25 public games (the speed-test shape: more games than
                         streams); a real submission is unaffected

    python scripts/_build_m2_vllm_kernel.py        # -> kaggle_submission_m2_vllm_s12/, _s14/
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from _build_m2_level_memory_kernel import cells_of, set_src, sub  # noqa: E402
from _m2_input_resolver import apply_input_resolver  # noqa: E402

INCUMBENT = ROOT / "kaggle_submission_m2_level_memory" / "notebook"
VLLM_SRC = ROOT / "kaggle_submission_m2_vllm"
SETUP_SRC = VLLM_SRC / "vendor" / "lordhansolo_vllm_setup.py"
SETUP_BODY_SHA256 = "59c57594d2c3a4792fc7bd96e0ae86ac19f213562dfd1ebd3b93162f0b384aa5"
LAUNCHER_SRC = VLLM_SRC / "launcher_cell.py"
SETUP_PATH = "/kaggle/arc3_vllm_setup.py"

VARIANTS = {"s12": 12, "s14": 14}
SERVED_MODEL_NAME = "flashnext"          # the name the pickled benchmark and cell 5 already use
LORD_MODEL_NAME = "primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8"
VLLM_PORT = 1234
RUNTIME_DATASET = "lordhansolo/vllm-main-e975732-arc3"
BUNDLE_DATASET = "lordhansolo/taaf-kaggle-source"
MODEL_SOURCE = "lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1"
HARNESS_BUNDLE = "dfranzen/taaf-kaggle-source-bundle-copy"
COMPETITION = "arc-prize-2026-arc-agi-3"
ARC_FRAME_SIDE = 64
DOCKER_PINNING = "original"   # kernel-metadata docker_image_pinning_type (same as the other m2 variant builders)


def kernel_slug(name: str) -> str:
    return f"arc3-m2-vllm-{name}"


def out_dir(name: str, root: Path = ROOT) -> Path:
    return root / f"kaggle_submission_m2_vllm_{name}" / "notebook"


def cell_index(src, needle):
    hits = [i for i, s in enumerate(src) if needle in s]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def incumbent_serving(src) -> dict:
    """What the incumbent's harness and SGLang launch assume: the server context length (window +
    8K headroom) and the board upscale, read from the incumbent cells rather than retyped."""
    launcher = src[cell_index(src, "PREFIX = \"/tmp/sgl-intel\"")]
    m = re.search(r"^\s*CTX=\(([\d+ ]+)\)\*1024,", launcher, re.M)
    assert m, "incumbent CTX not found"
    ctx = sum(int(x) for x in m.group(1).split("+")) * 1024
    env = src[cell_index(src, "'MULTIMODAL_UPSCALE':")]
    upscale = int(re.search(r"'MULTIMODAL_UPSCALE': '(\d+)'", env).group(1))
    window = re.search(r"'LOCAL_ANALYZER_CONTEXT_WINDOW': \(([\d+]+)\)\*1024", env)
    assert window and sum(int(x) for x in window.group(1).split("+")) * 1024 < ctx
    return {"max_model_len": ctx, "max_pixels": (ARC_FRAME_SIDE * upscale) ** 2}


# ---- lordhansolo's setup step: exact-anchor patches (each anchor must occur exactly once) ----
OVERLAY_GATE = r'''def arc3_overlay_mismatch(found: str, detail: str = '') -> None:
    """[calamitychasm] One greppable line, then stop, so a check run names a wrong overlay."""
    print(f'VLLM_OVERLAY_HASH_MISMATCH {found} {detail}'.rstrip(), flush=True)
    raise SystemExit(86)


def arc3_overlay_hash_gate(manifest: dict) -> None:
    """[calamitychasm] Hash the shipped overlay tar against the pinned identity before anything
    is unpacked, then run the stock identity/overlay validation under the same marker."""
    import hashlib
    expected = FINE_PREFIX_CACHE_PATCH_IDENTITY['overlay_sha256']
    patch = manifest.get('patch') or {}
    artifact = RUNTIME_DATASET / str(patch.get('artifact') or FINE_PREFIX_CACHE_PATCH_IDENTITY['artifact'])
    if not artifact.is_file():
        arc3_overlay_mismatch('missing', f'artifact={artifact}')
    digest = hashlib.sha256()
    with artifact.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    found = digest.hexdigest()
    if found != expected:
        arc3_overlay_mismatch(found, f'expected={expected} artifact={artifact.name}')
    if patch.get('overlay_sha256') != expected:
        arc3_overlay_mismatch(str(patch.get('overlay_sha256')), f'(runtime-manifest.json) expected={expected}')
    try:
        validate_prefix_cache_runtime(manifest)
    except RuntimeError as exc:
        arc3_overlay_mismatch('manifest-identity', ' '.join(str(exc).split()))
    print(f'VLLM_OVERLAY_HASH_OK {found}', flush=True)


def install_image_runtime() -> dict:
'''


def setup_patches(max_num_seqs: int, max_model_len: int, max_pixels: int) -> list[tuple[str, str, str]]:
    """(what, anchor, replacement) applied to the vendored setup step, in order."""
    tag = "  # [calamitychasm]"
    return [
        ("served model name",
         "SERVED_MODEL_NAME = 'primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8'\n",
         f"SERVED_MODEL_NAME = {SERVED_MODEL_NAME!r}{tag} the name the harness requests\n"),
        ("served model alias",
         "        '--served-model-name',\n        SERVED_MODEL_NAME,\n",
         f"        '--served-model-name',\n        SERVED_MODEL_NAME,\n        {LORD_MODEL_NAME!r},{tag} alias\n"),
        ("max model len",
         "VLLM_MAX_MODEL_LEN = 147072\n",
         f"VLLM_MAX_MODEL_LEN = {max_model_len}{tag} the incumbent's SGLang context length\n"),
        ("max pixels",
         "VLLM_IMAGE_MAX_PIXELS = 65536\n",
         f"VLLM_IMAGE_MAX_PIXELS = {max_pixels}{tag} 640x640 boards (MULTIMODAL_UPSCALE 10), never downscaled\n"),
        ("max num seqs",
         "VLLM_MAX_NUM_SEQS = 14\n",
         f"VLLM_MAX_NUM_SEQS = {max_num_seqs}{tag} = ARC3_MAX_ACTIVE_STREAMS\n"),
        ("images per prompt",
         "        json.dumps({'video': 0}),\n",
         "        json.dumps({'image': 64, 'video': 0}),"
         f"{tag} the harness attaches several boards (grid, diff, death, animation) per request\n"),
        ("chat template kwargs",
         """        '{"preserve_thinking": true, "reasoning_effort": "xhigh"}',\n""",
         """        '{"preserve_thinking": true}',""" + tag + " the incumbent's default; no reasoning_effort\n"),
        ("cached_tokens always reported",
         "    if os.environ.get('TAAF_RUN_AS_SUBMISSION') == '1':\n"
         "        cmd += ['--disable-log-stats', '--disable-uvicorn-access-log']\n"
         "    else:\n"
         "        cmd += ['--enable-prompt-tokens-details']\n",
         "    if os.environ.get('TAAF_RUN_AS_SUBMISSION') == '1':\n"
         "        cmd += ['--disable-log-stats', '--disable-uvicorn-access-log']\n"
         f"    cmd += ['--enable-prompt-tokens-details']{tag} usage reports cached_tokens in every run\n"),
        ("cache tracing opt-in",
         "    enabled = namespace['is_test_run']()\n",
         "    enabled = namespace['is_test_run']() and os.environ.get('ARC3_VLLM_CACHE_DIAGNOSTICS') == '1'"
         f"{tag} per-lookup tracing would bias the speed measurement\n"),
        ("overlay hash gate (definition)",
         "def install_image_runtime() -> dict:\n",
         OVERLAY_GATE),
        ("overlay hash gate (call)",
         "    manifest = read_runtime_manifest()\n    validate_prefix_cache_runtime(manifest)\n",
         "    manifest = read_runtime_manifest()\n    arc3_overlay_hash_gate(manifest)\n"),
        ("overlay applier failure",
         "        raise RuntimeError(f'Runtime patch failed:\\n{result.stderr.strip()}')\n",
         "        print(result.stderr.strip()[-4000:], flush=True)\n"
         "        found = re.search(r'has sha256 ([0-9a-f]{64})', result.stderr)\n"
         "        arc3_overlay_mismatch(found.group(1) if found else 'applier-failed', 'overlay applier rejected the runtime')\n"),
        ("re import",
         "import json\nimport math\n",
         "import json\nimport math\nimport re\n"),
    ]


SETUP_TAIL_START = "setup_env = {\n    'ARC3_CACHE_DIAGNOSTICS'"
SETUP_TAIL_END = "setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')\n"
SETUP_TAIL_NEW = ("# [calamitychasm] lordhansolo's harness env (his temperature, effort, 4x boards, ...) is not\n"
                  "# exported: the incumbent's cell 5 configures the harness.\n"
                  "print('VLLM_SETUP_DONE', flush=True)\n")


def patched_setup(max_num_seqs: int, max_model_len: int, max_pixels: int) -> str:
    text = SETUP_SRC.read_text(encoding="utf-8")
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != SETUP_BODY_SHA256:
        raise SystemExit(f"REFUSING TO BUILD -- vendored setup step changed (sha256 {got})")
    for what, old, new in setup_patches(max_num_seqs, max_model_len, max_pixels):
        text = sub(text, old, new, what)
    assert text.count(SETUP_TAIL_START) == 1 and text.count(SETUP_TAIL_END) == 1
    a = text.index(SETUP_TAIL_START)
    b = text.index(SETUP_TAIL_END) + len(SETUP_TAIL_END)
    return text[:a] + SETUP_TAIL_NEW + text[b:]


def launcher_source(max_num_seqs: int, max_model_len: int, max_pixels: int) -> str:
    text = LAUNCHER_SRC.read_text(encoding="utf-8")
    text = sub(text, "VLLM_MAX_NUM_SEQS = 14                 # [build]",
               f"VLLM_MAX_NUM_SEQS = {max_num_seqs}                 # [build]", "launcher seqs")
    text = sub(text, "VLLM_MAX_MODEL_LEN = 139264            # [build]",
               f"VLLM_MAX_MODEL_LEN = {max_model_len}            # [build]", "launcher len")
    text = sub(text, "VLLM_IMAGE_MAX_PIXELS = 409600         # [build]",
               f"VLLM_IMAGE_MAX_PIXELS = {max_pixels}         # [build]", "launcher pixels")
    text = sub(text, 'VLLM_SETUP_SCRIPT = Path("/kaggle/arc3_vllm_setup.py")',
               f"VLLM_SETUP_SCRIPT = Path({SETUP_PATH!r})", "setup path")
    return text


def env_cell(cell: str, max_num_seqs: int) -> str:
    """Cell 5: paths, port, reasoning echo key, stream count. Everything else stays."""
    old_paths = re.search(r"WHEELHOUSE_DIR    = '[^']*'\nMODEL_DIR         = '[^']*'\nDRAFT_MODEL_DIR   = '[^']*'\n", cell)
    assert old_paths, "incumbent model/wheelhouse paths not found"
    owner, model, framework, instance, version = MODEL_SOURCE.split("/")
    new_paths = (
        "# [calamitychasm vLLM] lordhansolo's runtime (vLLM e975732 image layers + overlay), his model\n"
        "# (with its built-in MTP head) and his bundle (only its 32k draft vocabulary is read).\n"
        f"VLLM_RUNTIME_DIR  = '/kaggle/input/datasets/{RUNTIME_DATASET}'\n"
        f"VLLM_MODEL_DIR    = '/kaggle/input/models/{owner}/{model}/{framework.lower()}/{instance}/{version}'\n"
        f"VLLM_BUNDLE_DIR   = '/kaggle/input/datasets/{BUNDLE_DATASET}'\n"
    )
    cell = cell.replace(old_paths.group(0), new_paths)
    cell = sub(cell, "SERVED_MODEL_PORT = 8001\n", f"SERVED_MODEL_PORT = {VLLM_PORT}   # [calamitychasm vLLM]\n", "port")
    cell = sub(cell, "'ARC3_REASONING_HISTORY_KEY': 'reasoning_content',",
               "'ARC3_REASONING_HISTORY_KEY': 'reasoning',   # [calamitychasm vLLM] vLLM's field", "reasoning key")
    cell = sub(cell, "'ARC3_MAX_ACTIVE_STREAMS': 10,",
               f"'ARC3_MAX_ACTIVE_STREAMS': {max_num_seqs},   # [calamitychasm vLLM] = --max-num-seqs", "streams")
    return cell


PRECACHE_TAIL = re.compile(r"precache_wheels_thread = threading\.Thread\(.*\Z", re.S)


def precache_cell(cell: str) -> str:
    m = PRECACHE_TAIL.search(cell)
    assert m and "precache_model_thread" in m.group(0)
    return cell[:m.start()] + (
        "# [calamitychasm vLLM] no Pennyroyal wheelhouse or dfranzen model to precache: lordhansolo's\n"
        "# watchdog runs a GPU shard prefetcher a bounded window ahead of each weight load.\n")


def all_public_games(cell: str) -> str:
    line = [ln for ln in cell.splitlines() if ln.startswith("demo_excluded_games = ")][0]
    return sub(cell, line, "demo_excluded_games = []   # [calamitychasm vLLM] all 25 public games (speed shape)",
               "games")


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(True)}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def build(name: str, root: Path = ROOT) -> Path:
    n = VARIANTS[name]
    nb = json.loads((INCUMBENT / "arc3-m2-level-memory.ipynb").read_text(encoding="utf-8"))
    src = cells_of(nb)
    knobs = incumbent_serving(src)

    i = cell_index(src, "'ARC3_REASONING_HISTORY_KEY'")
    set_src(nb, i, env_cell(src[i], n))
    i = cell_index(src, "def precache(")
    set_src(nb, i, precache_cell(src[i]))
    i = cell_index(src, "demo_excluded_games = [] if TRUE_SUBMISSION else")
    set_src(nb, i, all_public_games(src[i]))

    apply_input_resolver(nb)     # cell 5's path constants + the competition wheels; the launcher keeps its own
    src = cells_of(nb)
    launcher = cell_index(src, "PREFIX = \"/tmp/sgl-intel\"")
    assert nb["cells"][launcher - 1]["cell_type"] == "markdown" and "Start serving" in src[launcher - 1]
    nb["cells"][launcher - 1:launcher + 1] = [
        markdown("## 5. Start serving\n\n"
                 "[calamitychasm] vLLM instead of SGLang: lordhansolo's vLLM nightly e975732 runtime "
                 "(image layers + hash-pinned overlay, watchdog, shard prefetcher), his NVFP4/FP8 model with "
                 f"its built-in MTP head, {n} sequences. The first cell writes his setup step (patched only "
                 "for this harness); the second runs it and logs boot probes.\n"),
        code(f"%%writefile {SETUP_PATH}\n" + patched_setup(n, **knobs)),
        code(launcher_source(n, **knobs)),
    ]
    nb["cells"].insert(0, markdown(
        f"## [calamitychasm] {kernel_slug(name)}: incumbent harness on lordhansolo's vLLM stack\n\n"
        "The milestone-2 solution (`dfranzen/arc-agi-3-milestone-2-solution`, Apache-2.0; credit Daniel Franzen, "
        "Jeroen Cottaar, Tufa Labs) with our solved-level memory, served by the vLLM stack of "
        "`lordhansolo/arc-agi-3-milestone-2` (credit lordhansolo) instead of SGLang Pennyroyal. Harness cells "
        f"unchanged; serving: {n} streams, context {knobs['max_model_len']}, boards up to "
        f"{knobs['max_pixels']} px. The check run plays all 25 public games.\n"))

    d = out_dir(name, root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{kernel_slug(name)}.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps(metadata(name), indent=2), encoding="utf-8")
    return path


def metadata(name: str) -> dict:
    meta = json.loads((INCUMBENT / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(
        id=f"calamitychasm/{kernel_slug(name)}", title=kernel_slug(name),
        code_file=f"{kernel_slug(name)}.ipynb", is_private=True, enable_gpu=True, enable_internet=False,
        dataset_sources=[RUNTIME_DATASET, BUNDLE_DATASET, HARNESS_BUNDLE],
        model_sources=[MODEL_SOURCE], competition_sources=[COMPETITION],
    )
    # Keep the incumbent's pinned image (Python 3.12) and pin it explicitly: Kaggle's latest image moved to Python 3.13
    # by 2026-10-07 (JustAdev742 lesson 0029). The launcher's VLLM_PYTHON_ABI_MISMATCH preflight checks the vLLM
    # runtime's Python against the notebook's, so a runtime built for another Python fails loudly at boot.
    assert meta.get("docker_image", "").count("@sha256:") == 1, "the incumbent metadata must pin an image digest"
    meta["docker_image_pinning_type"] = DOCKER_PINNING
    return meta


def main(argv=None) -> int:
    from _check_notebook_cell import check_notebook
    for name in VARIANTS:
        path = build(name)
        problems = check_notebook(path)
        if problems:
            print("REFUSING TO SHIP", name, problems)
            return 1
        print("built", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

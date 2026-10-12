import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def is_test_run() -> bool:
    '''Return true only for an explicit noncompetition test run.'''

    return (
        os.environ.get('TAAF_RUN_AS_SUBMISSION', '').strip().lower() == '0'
        and os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'', '0', 'false'}
    )


RUNTIME_OWNER = 'lordhansolo'
RUNTIME_SLUG = 'vllm-main-e975732-arc3'
MODEL_SOURCE_PARTS = ('lordhansolo', 'qwen3-8-flash-next-mixed-nvfp4-fp8', 'PyTorch', 'hf-mixed-mtp-nvfp4', '1')
SERVED_MODEL_NAME = 'primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8'
VLLM_HOST = '127.0.0.1'
VLLM_PORT = 1234
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
VLLM_SERVER_URL = f'http://{VLLM_HOST}:{VLLM_PORT}'
VLLM_MAX_MODEL_LEN = 147072
ANALYZER_CONTEXT_WINDOW = 127488
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = '0.98'
VLLM_KV_CACHE_MEMORY_BYTES = 0
VLLM_MTP_TOKENS = 3
VLLM_CUDAGRAPH_CAPTURE_STEP = 8
WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
VLLM_SERVER_LOG = WORKING_DIR / 'vllm-openai-server.log'
VLLM_SERVER_PID = WORKING_DIR / 'vllm-openai-server.pid'
VLLM_SERVER_READY = WORKING_DIR / 'vllm-openai-server.ready'
VLLM_WATCHDOG_SCRIPT = WORKING_DIR / 'vllm-watchdog.py'
VLLM_WATCHDOG_CONFIG = WORKING_DIR / 'vllm-watchdog.json'
VLLM_WATCHDOG_LOG = WORKING_DIR / 'vllm-watchdog.log'
VLLM_WATCHDOG_PID = WORKING_DIR / 'vllm-watchdog.pid'
VLLM_SHARD_PREFETCH_REPORT = WORKING_DIR / 'vllm-shard-prefetch.json'
WATCHDOG_SCRIPT_TEXT = '\'\'\'Keep the vLLM OpenAI server alive for the whole harness run.\n\nStarted by the notebook setup step with the path of a JSON config holding the server command and\nthe server log, pid and ready-marker paths. The server is spawned as a child and awaited with a\nblocking wait, so a dead server is noticed without polling. Every exit is logged with its code,\nuptime and the tail of the server log. The server is restarted after a short pause, which lets the\nengine process that the dying API server kills release the GPU and the port. It is not restarted\nwhen it died before the setup step marked it ready, so a broken launch fails the setup step, and\nnot more than MAX_RESTARTS times, so a crash that comes back deterministically ends in a readable\n"giving up" line instead of a restart loop for the rest of the run.\n\nEach server start runs the GPU shard prefetcher from that start\'s log offset, so a restart never\nmistakes the previous load\'s completed progress bar for its own. Prefetch reports are kept\nseparately for the initial start and every restart. The prefetch runs in a separate process,\nwith bounded termination waits so a blocked NFS read cannot hold up server restarts.\n\'\'\'\nimport concurrent.futures\nimport json\nimport os\nimport re\nimport signal\nimport subprocess\nimport sys\nimport threading\nimport time\nfrom contextlib import contextmanager\nfrom datetime import datetime\nfrom pathlib import Path\n\n\ndef is_test_run() -> bool:\n    \'\'\'Return true only for an explicit noncompetition test run.\'\'\'\n\n    return (\n        os.environ.get(\'TAAF_RUN_AS_SUBMISSION\', \'\').strip().lower() == \'0\'\n        and os.environ.get(\'KAGGLE_IS_COMPETITION_RERUN\', \'\').strip().lower() in {\'\', \'0\', \'false\'}\n    )\n\n\nMAX_RESTARTS = 5\nRESTART_DELAY_SECONDS = 10\nHOST_MEMORY_SAMPLE_INTERVAL_SECONDS = 15\nGPU_SHARD_PREFETCH_THREADS = 4\nGPU_SHARD_PREFETCH_WINDOW_BYTES = 8 * 1024**3\nGPU_SHARD_PREFETCH_BLOCK_BYTES = 16 * 1024**2\nGPU_SHARD_PREFETCH_POLL_SECONDS = 0.5\nGPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS = 2\nPLE_SHARD_NAME_PREFIX = \'ple-\'\nGPU_SHARD_PREFETCH_ABORT = threading.Event()\nWATCHDOG_EXIT_DEFERRED = False\nWATCHDOG_EXIT_REQUESTED = False\n\nconfig = json.loads(Path(sys.argv[1]).read_text(encoding=\'utf-8\'))\nserver_command = config[\'command\']\nserver_log = Path(config[\'server_log\'])\nserver_pid = Path(config[\'server_pid\'])\nserver_ready = Path(config[\'server_ready\'])\n\n\ndef read_integer_file(path: str) -> int | None:\n    \'\'\'Read a cgroup counter, returning None for unreadable or nonnumeric values.\'\'\'\n    try:\n        return int(Path(path).read_text(encoding=\'ascii\').strip())\n    except (OSError, ValueError):\n        return None\n\n\ndef read_host_memory() -> dict[str, int | None]:\n    \'\'\'Report available RAM, including the cgroup limit used by vLLM\'s loader.\'\'\'\n    host_available_bytes = None\n    for line in Path(\'/proc/meminfo\').read_text(encoding=\'ascii\').splitlines():\n        if line.startswith(\'MemAvailable:\'):\n            host_available_bytes = int(line.split()[1]) * 1024\n            break\n    if host_available_bytes is None:\n        raise ValueError(\'MemAvailable is missing from /proc/meminfo\')\n\n    cgroup_limit_bytes = read_integer_file(\'/sys/fs/cgroup/memory.max\')\n    if cgroup_limit_bytes is not None:\n        cgroup_usage_bytes = read_integer_file(\'/sys/fs/cgroup/memory.current\')\n    else:\n        cgroup_limit_bytes = read_integer_file(\'/sys/fs/cgroup/memory/memory.limit_in_bytes\')\n        if cgroup_limit_bytes is not None and cgroup_limit_bytes >= 1 << 62:\n            cgroup_limit_bytes = None\n        cgroup_usage_bytes = (\n            read_integer_file(\'/sys/fs/cgroup/memory/memory.usage_in_bytes\')\n            if cgroup_limit_bytes is not None else None\n        )\n\n    cgroup_available_bytes = None\n    if cgroup_limit_bytes is not None:\n        cgroup_available_bytes = (\n            cgroup_limit_bytes\n            if cgroup_usage_bytes is None else max(0, cgroup_limit_bytes - cgroup_usage_bytes)\n        )\n    available_bytes = (\n        min(host_available_bytes, cgroup_available_bytes)\n        if cgroup_available_bytes is not None else host_available_bytes\n    )\n\n    return {\n        \'available_bytes\': available_bytes,\n        \'host_available_bytes\': host_available_bytes,\n        \'cgroup_available_bytes\': cgroup_available_bytes,\n        \'cgroup_limit_bytes\': cgroup_limit_bytes,\n        \'cgroup_usage_bytes\': cgroup_usage_bytes,\n    }\n\n\ndef log_host_memory() -> None:\n    \'\'\'Sample host RAM throughout server startup, inference, and restarts.\'\'\'\n    output_path = Path(config[\'host_memory_log\'])\n    try:\n        output_path.parent.mkdir(parents=True, exist_ok=True)\n        with output_path.open(\'w\', encoding=\'utf-8\') as output:\n            while True:\n                record = {\'ts\': round(time.time(), 3), **read_host_memory()}\n                output.write(json.dumps(record) + \'\\n\')\n                output.flush()\n                time.sleep(HOST_MEMORY_SAMPLE_INTERVAL_SECONDS)\n    except (OSError, ValueError) as error:\n        log_event(f\'host memory monitor stopped: {error!r}\')\n\n\ndef log_event(message: str) -> None:\n    print(f\'{datetime.now().isoformat(timespec="seconds")} watchdog: {message}\', flush=True)\n\n\ndef read_server_log_tail(lines: int = 40) -> str:\n    if not server_log.exists():\n        return \'\'\n\n    return \'\\n\'.join(server_log.read_text(encoding=\'utf-8\', errors=\'replace\').splitlines()[-lines:])\n\n\ndef build_natural_sort_key(path: Path) -> list:\n    \'\'\'Split a file name into text and integer parts, the key vLLM\'s safetensors loader sorts by.\'\'\'\n    return [int(part) if part.isdigit() else part for part in re.split(r\'(\\d+)\', path.name)]\n\n\ndef list_loader_shards(model_path: Path) -> list[Path]:\n    \'\'\'Return the checkpoint shards of the index in the order vLLM\'s safetensors loader reads them.\'\'\'\n    index = json.loads((model_path / \'model.safetensors.index.json\').read_text(encoding=\'utf-8\'))\n    shards = [model_path / name for name in set(index[\'weight_map\'].values())]\n\n    return sorted(shards, key=build_natural_sort_key)\n\n\ndef read_file_into_page_cache(path: Path, stop_event: threading.Event) -> dict:\n    \'\'\'Read a file in large blocks so its pages sit in the page cache, until the end or a stop.\n\n    The read stops early when its stop_event or GPU_SHARD_PREFETCH_ABORT is set. Both are checked\n    between blocks, so a stop takes effect only after an in-flight read returns. Returns the\n    monotonic start and end of the read and whether it was stopped early.\n    \'\'\'\n    started_at = time.monotonic()\n    buffer = memoryview(bytearray(GPU_SHARD_PREFETCH_BLOCK_BYTES))\n    with path.open(\'rb\', buffering=0) as handle:\n        while not (stopped := stop_event.is_set() or GPU_SHARD_PREFETCH_ABORT.is_set()) and handle.readinto(buffer):\n            pass\n\n    return {\'started_at\': started_at, \'finished_at\': time.monotonic(), \'stopped\': stopped}\n\n\ndef drop_file_from_page_cache(path: Path) -> None:\n    \'\'\'Ask the kernel to evict a file\'s clean cached pages. The kernel treats this as a hint.\n\n    An eviction that fails only leaves the pages in the cache, so it is reported and the prefetch\n    carries on with a window that is smaller in practice than it is on paper.\n    \'\'\'\n    try:\n        file_descriptor = os.open(path, os.O_RDONLY)\n        try:\n            os.posix_fadvise(file_descriptor, 0, 0, os.POSIX_FADV_DONTNEED)\n        finally:\n            os.close(file_descriptor)\n    except OSError as error:\n        print(f\'GPU shard prefetch could not evict {path.name}: {error!r}\', flush=True)\n\n\nclass LoaderProgressFollower:\n    \'\'\'Follow the completed-shard count of the target load\'s progress bar in the vLLM server log.\n\n    The bar prints `| N/<shard_count> [` after the N-th shard, one update per line. The MTP draft\'s\n    second bar has a different total, so it never matches. The unterminated tail of each read is\n    kept and completed by the next one, so a line cut by a read is still counted. The count is -1\n    until the bar\'s first line, which the loader prints once the pinned PLE table is allocated.\n    \'\'\'\n\n    def __init__(self, log_handle, shard_count: int) -> None:\n        self._log_handle = log_handle\n        self._pattern = re.compile(rf\'\\| (\\d+)/{shard_count} \\[\')\n        self._unterminated_text = \'\'\n        self.loaded_shard_count = -1\n\n    def update_loaded_shard_count(self) -> int:\n        \'\'\'Read the new log text and return the highest completed count seen so far.\'\'\'\n        complete_text, _, self._unterminated_text = (self._unterminated_text + self._log_handle.read()).rpartition(\'\\n\')\n        for match in self._pattern.finditer(complete_text):\n            self.loaded_shard_count = max(self.loaded_shard_count, int(match.group(1)))\n\n        return self.loaded_shard_count\n\n\nclass GpuShardPrefetcher:\n    \'\'\'Keep the GPU shards a bounded window ahead of vLLM\'s single-stream loader in the page cache.\n\n    The loader reads the GPU shards through one NFS stream at 44-153 MiB/s, while the PLE shards\n    already arrive at 615-923 MiB/s through its multi-threaded copy, so only the GPU shards are\n    prefetched. vLLM\'s own prefetch (v264, v265) read ahead with no bound, and after the pinned PLE\n    table the page cache holds about 20 GiB, so pages far ahead were evicted before the loader got\n    there. Nothing is read before the loader prints its first progress line, so the window never\n    competes with the PLE allocation. From then on shards are submitted in load order while the\n    bytes submitted and not yet passed by the loader fit the window, and the shard being loaded is\n    always submitted.\n\n    A shard the loader has passed gets its read stopped, and once that read has ended its pages\n    are evicted. Evicting while the read still runs would let it refill the cache behind the loader.\n    Pages read twice (the prefetch, then the loader) sit on the active LRU list and would\n    otherwise push the window\'s unread pages out first.\n\n    The loop returns once every GPU shard is passed and evicted, and writes a JSON report with the\n    per-shard prefetch and loader times, relative to the loader\'s first progress line. If the bar\n    never appears, nothing is read and no report is written. Shards the loader crossed inside one\n    poll carry the timestamp of the next printed update, so complete_before_loader_reached is an\n    upper bound and the report counts them. The log offset isolates each server start\'s progress.\n    \'\'\'\n\n    def __init__(self, shards: list[Path], server_log: Path, report_path: Path, log_offset: int = 0) -> None:\n        self._shards = shards\n        self._server_log = server_log\n        self._report_path = report_path\n        self._log_offset = log_offset\n        self._gpu_shard_indices = [\n            index for index, shard in enumerate(shards) if not shard.name.startswith(PLE_SHARD_NAME_PREFIX)\n        ]\n        self._shard_sizes = {index: shards[index].stat().st_size for index in self._gpu_shard_indices}\n        self._futures: dict[int, concurrent.futures.Future] = {}\n        self._stop_events: dict[int, threading.Event] = {}\n        self._progress_reached_at: dict[int, float] = {}\n        self._submitted_position = 0\n        self._passed_position = 0\n        self._dropped_position = 0\n\n    def run(self) -> None:\n        \'\'\'Follow the loader until every GPU shard is passed and evicted, then write the report.\n\n        GPU_SHARD_PREFETCH_ABORT ends the loop as well as the reads, and a pass that ends early\n        writes no report.\n        \'\'\'\n        while not self._server_log.exists():\n            if GPU_SHARD_PREFETCH_ABORT.is_set():\n                return\n            time.sleep(GPU_SHARD_PREFETCH_POLL_SECONDS)\n        with (\n            self._server_log.open(encoding=\'utf-8\', errors=\'replace\') as log_handle,\n            concurrent.futures.ThreadPoolExecutor(\n                max_workers=GPU_SHARD_PREFETCH_THREADS, thread_name_prefix=\'gpu-shard-prefetch\'\n            ) as executor,\n        ):\n            log_handle.seek(self._log_offset)\n            follower = LoaderProgressFollower(log_handle, len(self._shards))\n            while self._dropped_position < len(self._gpu_shard_indices) and not GPU_SHARD_PREFETCH_ABORT.is_set():\n                loaded_shard_count = follower.update_loaded_shard_count()\n                if loaded_shard_count >= 0:\n                    self._record_progress(loaded_shard_count)\n                    self._stop_passed_shards(loaded_shard_count)\n                    self._drop_passed_shards()\n                    self._submit_window(executor)\n                time.sleep(GPU_SHARD_PREFETCH_POLL_SECONDS)\n        if self._dropped_position == len(self._gpu_shard_indices) and self._progress_reached_at:\n            self._write_report()\n\n    def _record_progress(self, loaded_shard_count: int) -> None:\n        \'\'\'Stamp the first time each completed count was seen, including counts a poll skipped.\n\n        tqdm drops the updates that fall inside its minimum interval, so counts it never printed\n        carry the timestamp of the next printed one, which is later than the real crossing. Those\n        shards show equal reached and passed times in the report.\n        \'\'\'\n        now = time.monotonic()\n        for count in range(len(self._progress_reached_at), loaded_shard_count + 1):\n            self._progress_reached_at[count] = now\n\n    def _stop_passed_shards(self, loaded_shard_count: int) -> None:\n        \'\'\'Stop the reads of GPU shards the loader has finished and cancel those still queued.\'\'\'\n        while (\n            self._passed_position < len(self._gpu_shard_indices)\n            and self._gpu_shard_indices[self._passed_position] < loaded_shard_count\n        ):\n            shard_index = self._gpu_shard_indices[self._passed_position]\n            if shard_index in self._futures:\n                self._stop_events[shard_index].set()\n                self._futures[shard_index].cancel()\n            self._passed_position += 1\n\n    def _drop_passed_shards(self) -> None:\n        \'\'\'Evict passed GPU shards in order, each only after its read has ended.\'\'\'\n        while self._dropped_position < self._passed_position:\n            shard_index = self._gpu_shard_indices[self._dropped_position]\n            if shard_index in self._futures and not self._futures[shard_index].done():\n                return\n            drop_file_from_page_cache(self._shards[shard_index])\n            self._dropped_position += 1\n\n    def _submit_window(self, executor: concurrent.futures.ThreadPoolExecutor) -> None:\n        \'\'\'Submit the next GPU shards while the bytes still held in the page cache fit the window.\n\n        The window counts from the first shard not yet evicted, not from the loader, so shards the\n        loader has passed while their read winds down keep their bytes in the budget.\n        \'\'\'\n        self._submitted_position = max(self._submitted_position, self._passed_position)\n        ahead_bytes = sum(\n            self._shard_sizes[index]\n            for index in self._gpu_shard_indices[self._dropped_position:self._submitted_position]\n        )\n        while self._submitted_position < len(self._gpu_shard_indices) and (\n            self._submitted_position == self._passed_position\n            or ahead_bytes + self._shard_sizes[self._gpu_shard_indices[self._submitted_position]]\n            <= GPU_SHARD_PREFETCH_WINDOW_BYTES\n        ):\n            shard_index = self._gpu_shard_indices[self._submitted_position]\n            self._stop_events[shard_index] = threading.Event()\n            self._futures[shard_index] = executor.submit(\n                read_file_into_page_cache, self._shards[shard_index], self._stop_events[shard_index]\n            )\n            ahead_bytes += self._shard_sizes[shard_index]\n            self._submitted_position += 1\n\n    def _describe_shard(self, shard_index: int, loader_started_at: float) -> dict:\n        \'\'\'Return one shard\'s report row, with times in seconds after the loader\'s first progress line.\'\'\'\n        row = {\n            \'name\': self._shards[shard_index].name,\n            \'gib\': round(self._shard_sizes[shard_index] / 1024**3, 3),\n            \'loader_reached\': round(self._progress_reached_at[shard_index] - loader_started_at, 1),\n            \'loader_passed\': round(self._progress_reached_at[shard_index + 1] - loader_started_at, 1),\n        }\n        future = self._futures.get(shard_index)\n        if future is None or future.cancelled():\n            row[\'prefetch\'] = \'not started\'\n        elif future.exception() is not None:\n            row[\'prefetch\'] = f\'failed: {future.exception()!r}\'\n        else:\n            read = future.result()\n            row[\'prefetch\'] = \'stopped\' if read[\'stopped\'] else \'complete\'\n            row[\'prefetch_started\'] = round(read[\'started_at\'] - loader_started_at, 1)\n            row[\'prefetch_finished\'] = round(read[\'finished_at\'] - loader_started_at, 1)\n\n        return row\n\n    def _write_report(self) -> None:\n        \'\'\'Write the window settings and one row per GPU shard to the report file.\'\'\'\n        loader_started_at = self._progress_reached_at[0]\n        rows = [self._describe_shard(index, loader_started_at) for index in self._gpu_shard_indices]\n        report = {\n            \'threads\': GPU_SHARD_PREFETCH_THREADS,\n            \'window_gib\': GPU_SHARD_PREFETCH_WINDOW_BYTES / 1024**3,\n            \'gpu_shards\': len(self._gpu_shard_indices),\n            \'gpu_shard_gib\': round(sum(self._shard_sizes.values()) / 1024**3, 2),\n            \'other_shards\': len(self._shards) - len(self._gpu_shard_indices),\n            \'complete_before_loader_reached\': sum(\n                row[\'prefetch\'] == \'complete\' and row[\'prefetch_finished\'] <= row[\'loader_reached\'] for row in rows\n            ),\n            \'shards_crossed_within_one_poll\': sum(row[\'loader_reached\'] == row[\'loader_passed\'] for row in rows),\n            \'shards\': rows,\n        }\n        self._report_path.write_text(json.dumps(report, indent=1), encoding=\'utf-8\')\n\n\ndef prefetch_gpu_shards(log_offset: int, report_path: Path) -> None:\n    \'\'\'Build the GPU shard prefetcher for the mounted model and run it.\n\n    It runs in a separate process, so a failure while reading the checkpoint index\n    only ends the prefetch and leaves the server launch alone.\n    \'\'\'\n    GpuShardPrefetcher(\n        list_loader_shards(Path(config[\'model_path\'])), server_log, report_path, log_offset\n    ).run()\n\n\ndef start_gpu_shard_prefetch(log_offset: int, report_path: Path) -> subprocess.Popen | None:\n    \'\'\'Launch an isolated prefetch process, leaving the server running if spawning fails.\'\'\'\n    try:\n\n        return subprocess.Popen([\n            sys.executable, __file__, sys.argv[1], \'--prefetch\', str(log_offset), str(report_path)\n        ])\n    except OSError as error:\n        log_event(f\'could not start GPU shard prefetch: {error!r}\')\n\n        return None\n\n\ndef stop_gpu_shard_prefetch(prefetch_process: subprocess.Popen) -> None:\n    \'\'\'Terminate prefetch without waiting indefinitely for a blocked filesystem read.\'\'\'\n    if prefetch_process.poll() is not None:\n\n        return\n    prefetch_process.terminate()\n    try:\n        prefetch_process.wait(timeout=GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS)\n\n        return\n    except subprocess.TimeoutExpired:\n        log_event(f\'GPU shard prefetch did not stop within {GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS}s; sending SIGKILL\')\n    prefetch_process.kill()\n    try:\n        prefetch_process.wait(timeout=GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS)\n    except subprocess.TimeoutExpired:\n        log_event(\'GPU shard prefetch is still blocked after SIGKILL; continuing without waiting for it\')\n\n\n@contextmanager\ndef defer_watchdog_exit():\n    \'\'\'Finish registering or stopping prefetch before honoring a watchdog exit request.\'\'\'\n    global WATCHDOG_EXIT_DEFERRED\n    WATCHDOG_EXIT_DEFERRED = True\n    try:\n        yield\n    finally:\n        WATCHDOG_EXIT_DEFERRED = False\n        if WATCHDOG_EXIT_REQUESTED:\n            raise SystemExit(0)\n\n\ndef exit_watchdog(signal_number: int, frame) -> None:\n    \'\'\'Request shutdown once, deferring it while prefetch ownership changes or cleanup runs.\'\'\'\n    global WATCHDOG_EXIT_REQUESTED\n    if WATCHDOG_EXIT_REQUESTED:\n\n        return\n    WATCHDOG_EXIT_REQUESTED = True\n    if not WATCHDOG_EXIT_DEFERRED:\n        raise SystemExit(0)\n\n\ndef run_watchdog() -> None:\n    \'\'\'Spawn the server, wait for it to exit, log the exit and start it again while that is allowed.\'\'\'\n    restart_count = 0\n    prefetch_process = None\n    while True:\n        with server_log.open(\'a\' if restart_count else \'w\', encoding=\'utf-8\') as server_log_handle:\n            if restart_count:\n                server_log_handle.write(\n                    f\'\\n==== watchdog restart {restart_count} at {datetime.now().isoformat(timespec="seconds")} ====\\n\'\n                )\n                server_log_handle.flush()\n            log_offset = server_log_handle.tell()\n            started_at = time.monotonic()\n            server_process = subprocess.Popen(\n                server_command, stdout=server_log_handle, stderr=subprocess.STDOUT, text=True\n            )\n            server_pid.write_text(str(server_process.pid), encoding=\'utf-8\')\n            log_event(f\'vLLM server started (pid {server_process.pid}, restart {restart_count})\')\n            report_path = Path(config[\'shard_prefetch_report\'])\n            if restart_count:\n                report_path = report_path.with_name(f\'{report_path.stem}-restart-{restart_count}{report_path.suffix}\')\n            try:\n                with defer_watchdog_exit():\n                    if prefetch_process is None or prefetch_process.poll() is not None:\n                        prefetch_process = start_gpu_shard_prefetch(log_offset, report_path)\n                    else:\n                        log_event(\'previous GPU shard prefetch is still exiting; skipping another prefetch\')\n                exit_code = server_process.wait()\n                log_event(f\'vLLM server exited with code {exit_code} after {time.monotonic() - started_at:.0f}s\')\n            finally:\n                with defer_watchdog_exit():\n                    if prefetch_process is not None:\n                        stop_gpu_shard_prefetch(prefetch_process)\n        log_event(\'last server log lines:\\n\' + read_server_log_tail())\n        if not server_ready.exists():\n            log_event(\'server died before it was ready; not restarting\')\n            sys.exit(1)\n        if restart_count >= MAX_RESTARTS:\n            log_event(f\'giving up after {MAX_RESTARTS} restarts; the server stays down\')\n            sys.exit(1)\n        restart_count += 1\n        log_event(f\'restarting vLLM server in {RESTART_DELAY_SECONDS}s (restart {restart_count} of {MAX_RESTARTS})\')\n        time.sleep(RESTART_DELAY_SECONDS)\n\n\nif len(sys.argv) > 2 and sys.argv[2] == \'--prefetch\':\n    prefetch_gpu_shards(int(sys.argv[3]), Path(sys.argv[4]))\n    sys.exit(0)\nsignal.signal(signal.SIGTERM, exit_watchdog)\nif is_test_run():\n    threading.Thread(target=log_host_memory, name=\'host-memory-monitor\', daemon=True).start()\nrun_watchdog()\n'
CACHE_DIAGNOSTICS_SOURCE = '"""Test-only cache tracing, also installed as a standalone vLLM general plugin."""\n\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nimport logging\nimport os\nimport re\nimport time\nfrom pathlib import Path\n\n_ORIGINAL_LOOKUP = None\n_ORIGINAL_EVICT = None\n_ORIGINAL_SCHEDULE = None\n_FAILED = False\n\nENGINE_STEP_COLUMNS = (\n    "monotonic", "running", "decode_requests", "decode_tokens",\n    "prefill_requests", "prefill_tokens", "context_tokens",\n)\nENGINE_STEP_BATCH = 50\n_ENGINE_STEP_ROWS: list[list[float]] = []\n\n\ndef is_test_run(environ=None) -> bool:\n    environment = os.environ if environ is None else environ\n\n    return (\n        environment.get("TAAF_RUN_AS_SUBMISSION", "").strip().lower() == "0"\n        and environment.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()\n        in {"", "0", "false"}\n    )\n\n\ndef is_cache_diagnostics_enabled() -> bool:\n    return is_test_run() and os.environ.get("ARC3_CACHE_DIAGNOSTICS") == "1"\n\n\ndef hash_json(value) -> str:\n    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()\n\n\ndef encode_hash(value) -> str:\n    return value.hex() if isinstance(value, bytes) else str(value)\n\n\ndef append_cache_record(event: str, **fields) -> None:\n    global _FAILED\n    if _FAILED or not is_cache_diagnostics_enabled():\n        return\n    try:\n        directory = Path(os.environ["ARC3_CACHE_DIAGNOSTICS_DIR"])\n        directory.mkdir(parents=True, exist_ok=True)\n        path = directory / f"vllm_cache_diagnostics_{os.getpid()}.jsonl"\n        record = {"event": event, "ts": time.time(), "pid": os.getpid(), **fields}\n        with path.open("a", encoding="utf-8") as output:\n            output.write(json.dumps(record, separators=(",", ":")) + "\\n")\n    except Exception:\n        _FAILED = True\n        logging.getLogger(__name__).exception("Cache diagnostic writer disabled")\n\n\ndef summarize_group_blocks(coordinator) -> list[dict]:\n    """Count the pool blocks every KV cache group holds at this moment.\n\n    This is what separates a group\'s share of the fixed per-request cost from the\n    part that grows with context, which the lookup\'s own `free_blocks` only gives\n    as a total. Align-mode Mamba groups leave a null placeholder wherever a retired\n    state block used to sit, and several requests share one cached prefix block, so\n    both are collapsed by counting distinct non-null block ids rather than the\n    length of each request\'s block list. Summed over the groups these counts and\n    `free_blocks` have to account for `total_blocks`.\n    """\n    summary = []\n    for group_id, group_manager in enumerate(getattr(coordinator, "single_type_managers", ())):\n        block_ids = set()\n        requests = 0\n        for blocks in group_manager.req_to_blocks.values():\n            held = {block.block_id for block in blocks if not block.is_null}\n            block_ids |= held\n            requests += bool(held)\n        summary.append({\n            "group": group_id, "kind": type(group_manager.kv_cache_spec).__name__,\n            "block_size": group_manager.block_size, "requests": requests,\n            "blocks": len(block_ids),\n        })\n\n    return summary\n\n\ndef record_cache_lookup(manager, request, result) -> None:\n    started = time.monotonic()\n    coordinator = manager.coordinator\n    per_group = []\n    if manager.prefix_cache_lookup_enabled(request):\n        for spec, group_ids, manager_class, use_eagle in getattr(coordinator, "attention_groups", ()):\n            for drop_eagle in (False, True) if use_eagle else (False,):\n                _, hit = manager_class.find_longest_cache_hit(\n                    block_hashes=request.block_hashes,\n                    max_length=request.num_tokens - 1,\n                    kv_cache_group_ids=group_ids,\n                    block_pool=manager.block_pool,\n                    kv_cache_spec=spec,\n                    drop_eagle_block=drop_eagle,\n                    alignment_tokens=coordinator._cache_hit_alignment_tokens,\n                )\n                per_group.append({\n                    "groups": group_ids, "kind": type(spec).__name__,\n                    "block_size": spec.block_size, "drop_eagle": drop_eagle,\n                    "hit_tokens": hit,\n                })\n    tokens = request.prompt_token_ids or []\n    token_chunks = None\n    if not getattr(request, "_arc3_cache_prompt_logged", False):\n        token_chunks = [hash_json(tokens[offset:offset + 256]) for offset in range(0, len(tokens), 256)]\n        request._arc3_cache_prompt_logged = True\n    client_request = re.match(r"chatcmpl-arc3-[0-9a-f]{32}", request.request_id)\n    append_cache_record(\n        "cache_lookup", request_id=request.request_id,\n        client_request_id=client_request.group(0) if client_request else None,\n        prompt_tokens=len(tokens), computed_tokens=request.num_computed_tokens,\n        hit_tokens=result[1], shared_prefix_boundary=result[2],\n        hash_block_size=manager.block_pool.hash_block_size,\n        block_hashes=[encode_hash(value) for value in request.block_hashes],\n        token_chunk_size=256,\n        token_chunk_hashes=token_chunks,\n        per_group=per_group, kv_usage=manager.block_pool.get_usage(),\n        free_blocks=manager.block_pool.get_num_free_blocks(),\n        total_blocks=manager.block_pool.num_gpu_blocks,\n        group_blocks=summarize_group_blocks(coordinator),\n        prefix_caching_enabled=manager.enable_caching,\n        retention_interval=getattr(coordinator, "retention_interval", None),\n        diagnostic_seconds_before_write=time.monotonic() - started,\n    )\n\n\ndef trace_cache_lookup(manager, request):\n    global _FAILED\n    result = _ORIGINAL_LOOKUP(manager, request)\n    if is_cache_diagnostics_enabled() and not _FAILED:\n        try:\n            record_cache_lookup(manager, request, result)\n        except Exception:\n            _FAILED = True\n            logging.getLogger(__name__).exception("Cache lookup diagnostic failed")\n\n    return result\n\n\ndef trace_cache_eviction(pool, block):\n    global _FAILED\n    hashes = []\n    if is_cache_diagnostics_enabled() and not _FAILED:\n        try:\n            from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id\n\n            block_hashes = []\n            if block.block_hash is not None:\n                block_hashes.append(block.block_hash)\n            block_hashes.extend(pool.cached_block_hashes_by_block.get(block.block_id, ()))\n            hashes = [\n                {"hash": encode_hash(get_block_hash(value)), "group": get_group_id(value)}\n                for value in block_hashes\n            ]\n        except Exception:\n            _FAILED = True\n            logging.getLogger(__name__).exception("Cache eviction diagnostic failed")\n    result = _ORIGINAL_EVICT(pool, block)\n    if result and hashes:\n        append_cache_record("cache_eviction", block_id=block.block_id, hashes=hashes)\n\n    return result\n\n\ndef summarize_engine_step(scheduler, scheduler_output) -> list[float]:\n    """Describe one engine step with the batch composition its cost scales on.\n\n    A request is prefilling when the step started below the end of its prompt,\n    which also covers a final chunk as narrow as a decode. Context tokens sum\n    the computed tokens of every scheduled request and stand for the KV the step\n    reads, counted after the scheduler advanced them past this step.\n    """\n    decode_requests = 0\n    decode_tokens = 0\n    prefill_requests = 0\n    prefill_tokens = 0\n    context_tokens = 0\n    for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():\n        request = scheduler.requests[request_id]\n        context_tokens += request.num_computed_tokens\n        if request.num_computed_tokens - scheduled < request.num_prompt_tokens:\n            prefill_requests += 1\n            prefill_tokens += scheduled\n        else:\n            decode_requests += 1\n            decode_tokens += scheduled\n\n    return [\n        round(time.monotonic(), 6), len(scheduler.running),\n        decode_requests, decode_tokens, prefill_requests, prefill_tokens, context_tokens,\n    ]\n\n\ndef flush_engine_steps() -> None:\n    """Write the buffered steps, stamping the record right after the last row.\n\n    Nothing flushes a partial batch, so the batch bounds how much of a run\'s\n    tail a crash or a kill takes with it. The record\'s own wall-clock `ts`\n    belongs to its last row, which is what anchors these monotonic timestamps\n    against the wall-clock records of cache lookups.\n    """\n    if not _ENGINE_STEP_ROWS:\n        return\n    rows = list(_ENGINE_STEP_ROWS)\n    _ENGINE_STEP_ROWS.clear()\n    append_cache_record("engine_steps", columns=list(ENGINE_STEP_COLUMNS), rows=rows)\n\n\ndef trace_engine_step(scheduler, *args, **kwargs):\n    """Time steps by their spacing, which is what the engine sustains under load."""\n    global _FAILED\n    scheduler_output = _ORIGINAL_SCHEDULE(scheduler, *args, **kwargs)\n    if (\n        is_cache_diagnostics_enabled()\n        and not _FAILED\n        and scheduler_output.total_num_scheduled_tokens > 0\n    ):\n        try:\n            _ENGINE_STEP_ROWS.append(summarize_engine_step(scheduler, scheduler_output))\n            if len(_ENGINE_STEP_ROWS) >= ENGINE_STEP_BATCH:\n                flush_engine_steps()\n        except Exception:\n            _FAILED = True\n            logging.getLogger(__name__).exception("Engine step diagnostic failed")\n\n    return scheduler_output\n\n\ndef install_cache_diagnostics() -> None:\n    """Leave competition processes untouched, even if the plugin is present."""\n    global _ORIGINAL_LOOKUP, _ORIGINAL_EVICT, _ORIGINAL_SCHEDULE\n    if not is_cache_diagnostics_enabled() or _ORIGINAL_LOOKUP is not None:\n        return\n    try:\n        from vllm.v1.core.block_pool import BlockPool\n        from vllm.v1.core.kv_cache_manager import KVCacheManager\n        from vllm.v1.core.sched.scheduler import Scheduler\n\n        _ORIGINAL_LOOKUP = KVCacheManager.get_computed_blocks\n        _ORIGINAL_EVICT = BlockPool._maybe_evict_cached_block\n        _ORIGINAL_SCHEDULE = Scheduler.schedule\n        KVCacheManager.get_computed_blocks = trace_cache_lookup\n        BlockPool._maybe_evict_cached_block = trace_cache_eviction\n        Scheduler.schedule = trace_engine_step\n        append_cache_record("cache_diagnostics_installed")\n    except Exception:\n        logging.getLogger(__name__).exception("Cache diagnostics could not be installed")\n'
VLLM_QUANTIZATION = 'compressed-tensors'
VLLM_MOE_BACKEND = ''
VLLM_GDN_DECODE_KERNEL = 'triton'
VLLM_CUDA_LAUNCH_BLOCKING = False
VLLM_PROFILE_REQUESTED = False
VLLM_PROFILE_DIR = WORKING_DIR / 'vllm-profile'
# The image-layer runtime unpacks to ~19.5 GB. /kaggle/working is a 20 GB volume, while
# `df` inside a kernel on the RTX PRO 6000 host reported ~1.1 TB free on the root overlay,
# so the unpack goes to /tmp and the teardown removes it.
IMAGE_RUNTIME_ROOT = Path(os.environ.get('TAAF_IMAGE_RUNTIME_ROOT', '/tmp/vllm-image-runtime'))
IMAGE_RUNTIME_CACHE_ROOT = IMAGE_RUNTIME_ROOT.with_name(IMAGE_RUNTIME_ROOT.name + '-cache')
# The server.draft_vocab path relative to the bundled ARC3-Inference/ snapshot; empty keeps the
# full-vocabulary MTP draft head.
VLLM_DRAFT_VOCAB = 'configs/draft_vocab_32k.json'
VLLM_DRAFT_VOCAB_FILE = (
    Path(os.environ['TAAF_KAGGLE_BUNDLE_DIR']) / 'src' / 'ARC3-Inference' / VLLM_DRAFT_VOCAB
    if VLLM_DRAFT_VOCAB
    else None
)
if VLLM_DRAFT_VOCAB_FILE is not None and not VLLM_DRAFT_VOCAB_FILE.is_file():
    raise FileNotFoundError(f'Missing draft vocabulary: {VLLM_DRAFT_VOCAB_FILE}')
RUNTIME_MANIFEST_FILE_NAME = 'runtime-manifest.json'
VLLM_MAX_NUM_BATCHED_TOKENS = 2048
VLLM_DTYPE = 'bfloat16'
VLLM_KV_CACHE_DTYPE = 'fp8_e4m3'
VLLM_INDEXER_KV_DTYPE = 'fp8'
VLLM_MAMBA_SSM_CACHE_DTYPE = 'bfloat16'
VLLM_ENABLE_PREFIX_CACHING = True
VLLM_PREFIX_MATCH_UNIT = 128
VLLM_PREFIX_CACHE_RETENTION_INTERVAL = 0
# Pixel area of the agent's grid image, embedded from MULTIMODAL_UPSCALE at build time.
VLLM_IMAGE_MAX_PIXELS = 65536
FINE_PREFIX_CACHE_RUNTIME_IDENTITY = {
    'image': 'vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19',
    'amd64_manifest_digest': 'sha256:e0eee5c5506bea9bfe350f7d99b07dc49e37d42647a128c2a57ff184551fba10',
    'vllm_version': '0.29.1rc1.dev573+ge97573215',
}
FINE_PREFIX_CACHE_PATCH_IDENTITY = {
    'artifact': 'arc3_vllm_main_e975732_arc3_overlay.tar.blob',
    'applier': 'apply_vllm_main_e975732_arc3.py',
    'overlay_sha256': 'e3a6fe0d9f010bc1fb66e53e7dc6fde43272824d5200da5a49b5c4456065e0cb',
    'vllm_commit': 'e9757321527ca1ecd514c07c1418dd2c53da3d19',
}
FINE_PREFIX_CACHE_REQUIRED_OVERLAYS = {
    'vllm/v1/core/kv_cache_coordinator.py': {
        'stock_sha256': '7fa4065d19021e77b3424d24e1b088c530ed70b90c5ed9bc52cf4b7581d5fc48',
        'sha256': '7494e9f76d4a6b0bb469e80816e4052fb52171bc63b34799c3b9588092bc243e',
    },
    'vllm/v1/core/kv_cache_utils.py': {
        'stock_sha256': 'd359221ef91a94f7e570220f083979297c297227c1ad1b8ccdbec76cc24565b7',
        'sha256': '816aea33eaa28a15b8d739be602a877aa0be1a8ec83853cdc92412dcdd137b4e',
    },
    'vllm/v1/core/single_type_kv_cache_manager.py': {
        'stock_sha256': '6ebd4ddb5210d500b52b300dfaf462e2e92f58a4ee0e15691a860549dc086d68',
        'sha256': '97a0a4518c757cbe48e0d13d7a1890f48bc0db777004a3d66a8eb9ea73ba2405',
    },
    'vllm/v1/worker/gpu/model_states/mamba_hybrid.py': {
        'stock_sha256': '6ea89adcb39762b533f0c5b765a66e6f942cbde90208a5d13a758ef86a4ac6f1',
        'sha256': '8cd9eac0da31add7685ba7f23dfc2dcd7d6d422470fa98281aa73f90d5a8408d',
    },
    'vllm/v1/worker/gpu_model_runner.py': {
        'stock_sha256': '95dbd301feae580913938cafe4151da7af2cbd5d3f37fa645fab62a8cd1b54ef',
        'sha256': 'a2d3709963d463256b8b62ecb5264bce10e1b133d552ab10afdd31f68b42d1b2',
    },
    'vllm/v1/worker/mamba_utils.py': {
        'stock_sha256': '28c8a9bb07a56e3612e245ccc48cd7f29d85181dbe5e4cf8940c6bcd086df856',
        'sha256': 'b5e30c8ab2343340f478c9c8ea94b9369649c74044cd42ea11b246dc84e11243',
    },
}
# Embedded from KAGGLE_VLLM_MAX_NUM_SEQS at build time, which the launcher fills from
# server.max_num_seqs, the way the KV pin and the MTP tokens are filled.
VLLM_MAX_NUM_SEQS = 14
# The deadline covers layer unpack, host PLE allocation, both target and MTP weight loads,
# engine init and CUDA graph capture. The mixed checkpoint's 167.7 GiB safetensors set is larger
# than the free RAM left after allocating the BF16 PLE table, so an NFS cold start can take about
# 30 minutes. Keep enough margin for Kaggle storage variance while still detecting a stuck start.
VLLM_SERVER_READY_TIMEOUT_SECONDS = 3600

GPU_NAME_PATTERNS = {'rtx-pro-6000': ('rtx pro 6000',), 'h100': ('h100',), 'l4': ('l4',)}


def taaf_kaggle_input_paths() -> dict[str, Path]:
    raw = os.getenv('TAAF_KAGGLE_INPUT_PATHS', '').strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError('TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.')
    return {str(ref): Path(str(path)) for ref, path in data.items()}


def resolve_kaggle_dataset_path(owner: str, slug: str) -> Path:
    mapped = taaf_kaggle_input_paths().get(f'{owner}/{slug}')
    if mapped is not None:
        return mapped
    for dataset_path in (Path('/kaggle/input') / slug, Path('/kaggle/input/datasets') / owner / slug):
        if dataset_path.exists():
            return dataset_path
    return Path('/kaggle/input') / slug


def resolve_kaggle_model_path(parts: tuple[str, str, str, str, str]) -> Path:
    owner, model, framework, instance, version = parts
    candidates = []
    for framework_name in dict.fromkeys((framework.lower(), framework)):
        candidates.append(Path('/kaggle/input/models') / owner / model / framework_name / instance / version)
        candidates.append(Path('/kaggle/input') / model / framework_name / instance / version)
    for model_path in candidates:
        if model_path.exists():
            return model_path

    return candidates[0]


def describe_kaggle_input_tree(max_depth: int = 5) -> str:
    root = Path('/kaggle/input')
    if not root.exists():
        return 'No /kaggle/input mount is present.'
    lines = []
    frontier = [root]
    for _ in range(max_depth):
        children = []
        for directory in frontier:
            for entry in sorted(directory.iterdir()):
                if entry.is_dir():
                    lines.append('  ' + str(entry))
                    children.append(entry)
        frontier = children

    return 'Directories under /kaggle/input:\n' + '\n'.join(lines)


RUNTIME_DATASET = resolve_kaggle_dataset_path(RUNTIME_OWNER, RUNTIME_SLUG)
MODEL_PATH = resolve_kaggle_model_path(MODEL_SOURCE_PARTS)


def assert_expected_cuda_gpu() -> None:
    if not Path('/kaggle/input').exists():
        return
    assert shutil.which('nvidia-smi'), 'CUDA GPU check failed: nvidia-smi is not available.'
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    assert result.returncode == 0, f'nvidia-smi failed: {result.stderr.strip()}'
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert gpu_names, 'nvidia-smi did not report any CUDA GPUs.'
    expected_gpu_type = os.getenv('KAGGLE_GPU_TYPE', 'rtx-pro-6000').strip().lower()
    expected_count = os.getenv('KAGGLE_GPU_COUNT', '1')
    if expected_count.isdigit():
        assert len(gpu_names) == int(expected_count), f'Expected {expected_count} CUDA GPU(s), found {gpu_names}'
    patterns = GPU_NAME_PATTERNS.get(expected_gpu_type, (expected_gpu_type.replace('-', ' '),))
    mismatched = [name for name in gpu_names if not any(pattern in name.lower() for pattern in patterns)]
    assert not mismatched, f'Expected GPU type {expected_gpu_type!r}, found {gpu_names}'
    print(f'CUDA GPU check passed for {expected_gpu_type} x{expected_count}: {gpu_names}', flush=True)


def prepend_path_entries(existing: str, entries: list[Path]) -> str:
    joined = os.pathsep.join([str(entry) for entry in entries] + ([existing] if existing else []))

    return joined


def read_runtime_manifest() -> dict:
    manifest_path = RUNTIME_DATASET / RUNTIME_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing image runtime manifest: {manifest_path}')

    return json.loads(manifest_path.read_text(encoding='utf-8'))


def build_runtime_python_paths(manifest: dict) -> list[Path]:
    '''PYTHONPATH entries; the cutlass DSL .pth of the image is not processed on PYTHONPATH, so add its target.'''
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']

    return [dist_packages / 'nvidia_cutlass_dsl' / 'dsl_packages', dist_packages]


def resolve_runtime_cuda_home(manifest: dict) -> Path:
    cuda_home = IMAGE_RUNTIME_ROOT / Path(manifest['nvcc']).parent.parent

    return cuda_home


def build_runtime_library_paths(manifest: dict) -> list[Path]:
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    nvidia_libs = sorted(dist_packages.glob('nvidia/*/lib')) + sorted(dist_packages.glob('nvidia/cu13/*/lib'))

    return [
        resolve_runtime_cuda_home(manifest) / 'targets' / 'x86_64-linux' / 'lib',
        dist_packages / 'torch' / 'lib',
        *nvidia_libs,
        Path('/usr/local/nvidia/lib64'),
    ]


def build_image_runtime_env(manifest: dict) -> dict[str, str]:
    '''Build the vLLM server environment for the extracted image runtime.

    With the b12x MoE backend the b12x micro path is disabled, so every batch runs through
    the dynamic kernel. vLLM marks padding rows with expert id -1, and the b12x 1.3.0 micro
    kernel (up to 6 tokens at top-10) indexes weights with that id, which crashes with an
    illegal memory access. The dynamic kernel skips such rows.

    A configured draft vocabulary is read in place from the mounted bundle and handed to the
    e975732 rc3 runtime through VLLM_QWEN4_EXP_DRAFT_VOCAB; without one the variable is removed.

    VLLM_ARC3_GDN_RECOVERSSM=1 makes the GDN layers verify MTP drafts without per-draft state
    blocks and commit the accepted tokens after sampling, which frees 9 KV blocks per request.
    It is only a default, so an A/B launcher that sets the variable to 0 keeps the stock path.
    '''
    env = os.environ.copy()
    cuda_home = resolve_runtime_cuda_home(manifest)
    cache_paths = {
        'VLLM_CACHE_ROOT': IMAGE_RUNTIME_CACHE_ROOT / 'vllm',
        'TRITON_CACHE_DIR': IMAGE_RUNTIME_CACHE_ROOT / 'triton',
        'FLASHINFER_WORKSPACE_BASE': IMAGE_RUNTIME_CACHE_ROOT / 'flashinfer',
        'B12X_COMPILE_CACHE_DIR': IMAGE_RUNTIME_CACHE_ROOT / 'b12x' / 'compile',
        'TMPDIR': IMAGE_RUNTIME_CACHE_ROOT / 'tmp',
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    env.pop('PYTORCH_ALLOC_CONF', None)
    # These legacy variables are either deprecated or no longer consumed by the pinned runtime.
    # Their supported equivalents are passed by build_vllm_server_command.
    env.pop('VLLM_PLE_CPU_OFFLOAD', None)
    env.pop('VLLM_PLE_OFFLOAD_READY_TIMEOUT', None)
    env.pop('VLLM_PREFIX_CACHE_RETENTION_INTERVAL', None)
    env['PYTHONPATH'] = prepend_path_entries(env.get('PYTHONPATH', ''), build_runtime_python_paths(manifest))
    env['PATH'] = prepend_path_entries(env.get('PATH', ''), [cuda_home / 'bin', IMAGE_RUNTIME_ROOT / 'usr' / 'local' / 'bin'])
    env['LD_LIBRARY_PATH'] = prepend_path_entries(
        env.get('LD_LIBRARY_PATH', ''), [path for path in build_runtime_library_paths(manifest) if path.is_dir()]
    )
    env.update({key: str(path) for key, path in cache_paths.items()})
    env.update(
        {
            'USE_TF': '0',
            'TRANSFORMERS_NO_TF': '1',
            'TRANSFORMERS_NO_TORCHVISION': '1',
            'VLLM_NO_USAGE_STATS': '1',
            'CUDA_HOME': str(cuda_home),
            'CUDACXX': str(cuda_home / 'bin' / 'nvcc'),
            'TORCH_CUDA_ARCH_LIST': '12.0',
            'VLLM_ENABLE_CUDA_COMPATIBILITY': '0',
            'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
            'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False',
            'HF_HUB_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false',
        }
    )
    env.setdefault('VLLM_ARC3_GDN_RECOVERSSM', '1')
    if VLLM_GDN_DECODE_KERNEL:
        env['VLLM_GDN_DECODE_KERNEL'] = VLLM_GDN_DECODE_KERNEL
    if VLLM_MOE_BACKEND == 'b12x':
        env['B12X_MICRO_DYNAMIC_CUTOVER_PAIRS'] = '0'
    if VLLM_CUDA_LAUNCH_BLOCKING:
        env['CUDA_LAUNCH_BLOCKING'] = '1'
    env.pop('VLLM_QWEN4_EXP_DRAFT_VOCAB', None)
    if VLLM_DRAFT_VOCAB_FILE is not None:
        env['VLLM_QWEN4_EXP_DRAFT_VOCAB'] = str(VLLM_DRAFT_VOCAB_FILE)

    return env


def verify_layer_blob(layer: dict) -> Path:
    '''Kaggle already guarantees dataset integrity, so only the recorded size is checked here.'''
    blob = RUNTIME_DATASET / layer['file']
    if not blob.exists():
        raise FileNotFoundError(f'Missing runtime layer blob: {blob}')
    if blob.stat().st_size != int(layer['size']):
        raise RuntimeError(f'Runtime layer {layer["index"]} does not match its manifest size: {blob.name}')

    return blob


def clear_opaque_whiteout_targets(layer: dict) -> None:
    for whiteout in layer.get('whiteouts', []):
        if whiteout['kind'] == 'opaque' and whiteout['target'] not in ('', '.'):
            shutil.rmtree(IMAGE_RUNTIME_ROOT / whiteout['target'], ignore_errors=True)


def remove_whiteout_markers_and_targets(layer: dict) -> None:
    for whiteout in layer.get('whiteouts', []):
        (IMAGE_RUNTIME_ROOT / whiteout['marker']).unlink(missing_ok=True)
        if whiteout['kind'] == 'file':
            target = IMAGE_RUNTIME_ROOT / whiteout['target']
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)


def extract_layer(blob: Path) -> None:
    '''pigz, when the image ships it, moves reading, checksumming and writing off the single inflate thread
    (roughly 1.2-1.5x over gzip); the .pyc files stay in so the server does not recompile vLLM and torch.'''
    decompressor = ['--use-compress-program=pigz'] if shutil.which('pigz') else ['-z']
    subprocess.run(
        ['tar', '-xf', str(blob), '-C', str(IMAGE_RUNTIME_ROOT), '--no-same-owner', *decompressor],
        check=True,
    )


def apply_runtime_patch(manifest: dict) -> None:
    patch = manifest.get('patch')
    if not patch:
        print('Image runtime ships no patch', flush=True)

        return
    overlay_files = patch.get('overlay_files')
    if not isinstance(overlay_files, list) or not overlay_files:
        raise RuntimeError('Runtime patch requires a non-empty overlay_files manifest')
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    result = subprocess.run(
        [sys.executable, str(RUNTIME_DATASET / patch['applier']), str(dist_packages)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f'Runtime patch failed:\n{result.stderr.strip()}')
    print(f'Runtime patch {patch["artifact"]}: {result.stdout.strip()}', flush=True)


def install_image_runtime() -> dict:
    '''Unpack the image layers in index order, replay whiteouts and apply the runtime overlay.'''
    manifest = read_runtime_manifest()
    validate_prefix_cache_runtime(manifest)
    shutil.rmtree(IMAGE_RUNTIME_ROOT, ignore_errors=True)
    IMAGE_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    layers = sorted(manifest['selected_layers'], key=lambda layer: int(layer['index']))
    started = time.monotonic()
    for layer in layers:
        blob = verify_layer_blob(layer)
        clear_opaque_whiteout_targets(layer)
        extract_layer(blob)
        remove_whiteout_markers_and_targets(layer)
        print(f'Extracted runtime layer {layer["index"]} ({blob.name}) after {time.monotonic() - started:.0f}s', flush=True)
    apply_runtime_patch(manifest)
    print(
        f'Image runtime ready at {IMAGE_RUNTIME_ROOT}: vLLM {manifest["vllm_version"]}, torch {manifest["torch_version"]}',
        flush=True,
    )

    return manifest


def validate_prefix_cache_runtime(manifest: dict) -> None:
    '''Fail closed when fine-grained Qwen3.8 caching lacks its Mamba/MTP safety overlays.'''
    if not VLLM_ENABLE_PREFIX_CACHING or VLLM_PREFIX_MATCH_UNIT <= 0:

        return

    identity_mismatches = sorted(
        key
        for key, expected_value in FINE_PREFIX_CACHE_RUNTIME_IDENTITY.items()
        if manifest.get(key) != expected_value
    )
    if identity_mismatches:
        raise RuntimeError(
            'Fine-grained Qwen3.8 prefix caching requires the pinned vLLM runtime identity; '
            'mismatched fields: ' + ', '.join(identity_mismatches)
        )

    patch = manifest.get('patch') or {}
    patch_identity_mismatches = sorted(
        key
        for key, expected_value in FINE_PREFIX_CACHE_PATCH_IDENTITY.items()
        if patch.get(key) != expected_value
    )
    if patch_identity_mismatches:
        raise RuntimeError(
            'Fine-grained Qwen3.8 prefix caching requires the pinned runtime patch identity; '
            'mismatched fields: ' + ', '.join(patch_identity_mismatches)
        )

    overlays = patch.get('overlay_files', [])
    overlay_targets = [overlay.get('target') for overlay in overlays]
    overlays_by_target = {
        overlay.get('target'): overlay
        for overlay in overlays
    }
    missing_targets = sorted(
        set(FINE_PREFIX_CACHE_REQUIRED_OVERLAYS) - set(overlay_targets)
    )
    duplicate_targets = sorted(
        target
        for target in FINE_PREFIX_CACHE_REQUIRED_OVERLAYS
        if overlay_targets.count(target) > 1
    )
    mismatched_targets = sorted(
        target
        for target, expected_hashes in FINE_PREFIX_CACHE_REQUIRED_OVERLAYS.items()
        if target in overlays_by_target
        and any(
            overlays_by_target[target].get(hash_name) != expected_hash
            for hash_name, expected_hash in expected_hashes.items()
        )
    )
    problems = []
    if missing_targets:
        problems.append('missing targets: ' + ', '.join(missing_targets))
    if duplicate_targets:
        problems.append('duplicate targets: ' + ', '.join(duplicate_targets))
    if mismatched_targets:
        problems.append('mismatched overlay hashes: ' + ', '.join(mismatched_targets))
    if problems:
        raise RuntimeError(
            'Fine-grained Qwen3.8 prefix caching requires the validated Mamba/MTP runtime overlays; '
            + '; '.join(problems)
        )


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def request_no_content(url: str, timeout: int = 30) -> None:
    request = urllib.request.Request(url, data=b'', method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f'Unexpected HTTP {response.status} from {url}')


def tail_log(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ''
    return '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])


def wait_for_vllm_server(watchdog_process: subprocess.Popen, timeout_seconds: int) -> None:
    '''Poll the models endpoint until the server answers.

    The watchdog exits when the server dies before it was marked ready, and polling the watchdog
    reaps it, so a broken launch raises at once instead of running out the timeout.
    '''
    deadline = time.monotonic() + timeout_seconds
    url = f'{VLLM_BASE_URL}/models'
    while time.monotonic() < deadline:
        if watchdog_process.poll() is not None:
            raise RuntimeError(
                f'vLLM watchdog exited with code {watchdog_process.returncode} before the server was ready.\n'
                f'{tail_log(VLLM_WATCHDOG_LOG)}'
            )
        try:
            models = request_json(url, timeout=5)
            print('vLLM server ready:', models, flush=True)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f'Timed out waiting for vLLM server at {url}.\nLast server log lines:\n{tail_log(VLLM_SERVER_LOG)}')


def build_vllm_server_command() -> list[str]:
    '''The Qwen3.8-Flash-Next launch on the packed image runtime, as verified on Kaggle in v187.

    Prefix caching follows the config. With it enabled, the hybrid model uses
    mamba_cache_mode "align" and the runtime aligns Mamba checkpoints to the resolved
    attention block size. The patched runtime is required for correct scheduler and
    worker alignment; saved checkpoints share the KV pool with active requests.

    The quantization method, the sequence cap and the KV cache dtype come from the config, so all
    three reach this command. The method is passed explicitly rather than left to vLLM's detection,
    so a checkpoint whose config.json disagrees fails at startup instead of loading under a scheme
    nobody picked. The cap has to follow the KV pool rather than the worker count, because a cap
    above what the pool holds only queues the surplus in the scheduler. The experimental runtime
    adds FP8 KV support to QSA attention. Auto is passed as no flag at all; FP8 is enabled only
    when explicitly selected in the config. The MTP head is the model's own drafter, so no
    draft model rides along. The pinned KV cache size is optional; zero lets vLLM size the pool
    from gpu_memory_utilization after profiling the activation peak.

    A decode step carries one token per sequence plus its MTP drafts, and vLLM captures CUDA
    graphs at 1, 2 and 4 tokens and then at every multiple of 8 below 256. A capture size off that
    ladder is truncated down to the last size it captured, so a full batch whose step lands above
    it decodes with no graph at all. That is what nine sequences did in v216, a 36-token step
    against a largest graph of 32. The size is rounded up to the next multiple of 8 instead, and a
    step that falls between two sizes pads into the graph above it.

    With a draft vocabulary the MTP drafter picks its greedy proposals through
    use_local_argmax_reduction, which the e975732 rc3 runtime serves from the listed LM-head rows
    only. The target still verifies against the full head, so the list changes draft acceptance
    and never the generated text.

    embed_tokens in --cpu-offload-params keeps the 1.18 GiB input embedding table in pinned host
    memory, next to the PLE table, and the KV pool gets those bytes. The e975732 rc3 runtime reads
    only the requested rows over PCIe, so embeddings stay bit-identical, and the MTP drafter shares
    the table. The table sits outside the --cpu-offload-gb budget, which stays 0, so vLLM's generic
    weight offloader never starts.

    vLLM profiles the vision encoder on the largest image the preprocessor accepts, 4096x4096
    or 16384 tokens, and in v257 that one dummy set the whole 2.19 GiB activation peak, which the
    KV pool gives up. The agent only ever sends its grid image, so max_pixels caps the
    preprocessor at that image's area. The profile then encodes as many grid images as one
    max_num_batched_tokens chunk holds, the worst case a real step can meet, while a grid image is
    processed exactly as before. Video is disabled, otherwise its 12288-token dummy becomes the profiled item.
    '''
    max_num_seqs = VLLM_MAX_NUM_SEQS
    decode_step_tokens = max_num_seqs * (1 + VLLM_MTP_TOKENS)
    cudagraph_capture_size = (
        math.ceil(decode_step_tokens / VLLM_CUDAGRAPH_CAPTURE_STEP) * VLLM_CUDAGRAPH_CAPTURE_STEP
    )
    cmd = [
        sys.executable,
        '-m',
        'vllm.entrypoints.cli.main',
        'serve',
        str(MODEL_PATH),
        '--served-model-name',
        SERVED_MODEL_NAME,
        '--host',
        VLLM_HOST,
        '--port',
        str(VLLM_PORT),
        '--load-format',
        'safetensors',
        '--dtype',
        VLLM_DTYPE,
        '--tensor-parallel-size',
        str(VLLM_TENSOR_PARALLEL_SIZE),
        '--distributed-executor-backend',
        'mp',
        '--max-model-len',
        str(VLLM_MAX_MODEL_LEN),
        '--max-num-seqs',
        str(max_num_seqs),
        '--max-num-batched-tokens',
        str(VLLM_MAX_NUM_BATCHED_TOKENS),
        '--async-scheduling',
        '--enable-chunked-prefill',
        '--max-cudagraph-capture-size',
        str(cudagraph_capture_size),
        '--enable-prefix-caching' if VLLM_ENABLE_PREFIX_CACHING else '--no-enable-prefix-caching',
        '--enable-auto-tool-choice',
        '--tool-call-parser',
        'qwen3_coder',
        '--reasoning-parser',
        'qwen3',
        '--generation-config',
        'vllm',
        '--engram-config',
        json.dumps({'cpu_offload': True}),
        '--cpu-offload-params',
        'embed_tokens',
        '--mm-processor-kwargs',
        json.dumps({'max_pixels': VLLM_IMAGE_MAX_PIXELS}),
        '--limit-mm-per-prompt',
        json.dumps({'video': 0}),
        '--default-chat-template-kwargs',
        '{"preserve_thinking": true, "reasoning_effort": "xhigh"}',
    ]
    if VLLM_QUANTIZATION:
        cmd += ['--quantization', VLLM_QUANTIZATION]
    if VLLM_MOE_BACKEND:
        cmd += ['--moe-backend', VLLM_MOE_BACKEND]
    if VLLM_ENABLE_PREFIX_CACHING and VLLM_PREFIX_MATCH_UNIT > 0:
        cmd += ['--prefix-match-unit', str(VLLM_PREFIX_MATCH_UNIT)]
    if VLLM_ENABLE_PREFIX_CACHING and VLLM_PREFIX_CACHE_RETENTION_INTERVAL >= 0:
        cmd += [
            '--prefix-cache-retention-interval',
            str(VLLM_PREFIX_CACHE_RETENTION_INTERVAL),
        ]
    if VLLM_MTP_TOKENS > 0:
        speculative_config = {'method': 'mtp', 'num_speculative_tokens': VLLM_MTP_TOKENS}
        if VLLM_DRAFT_VOCAB:
            speculative_config['use_local_argmax_reduction'] = True
        cmd += ['--speculative-config', json.dumps(speculative_config)]
    if VLLM_GPU_MEMORY_UTILIZATION:
        cmd += ['--gpu-memory-utilization', VLLM_GPU_MEMORY_UTILIZATION]
    if VLLM_KV_CACHE_MEMORY_BYTES > 0:
        cmd += ['--kv-cache-memory-bytes', str(VLLM_KV_CACHE_MEMORY_BYTES)]
    if VLLM_KV_CACHE_DTYPE and VLLM_KV_CACHE_DTYPE != 'auto':
        cmd += ['--kv-cache-dtype', VLLM_KV_CACHE_DTYPE]
    if VLLM_INDEXER_KV_DTYPE:
        cmd += ['--attention-config', json.dumps({'indexer_kv_dtype': VLLM_INDEXER_KV_DTYPE})]
    if VLLM_MAMBA_SSM_CACHE_DTYPE and VLLM_MAMBA_SSM_CACHE_DTYPE != 'auto':
        cmd += ['--mamba-ssm-cache-dtype', VLLM_MAMBA_SSM_CACHE_DTYPE]
    chat_template = MODEL_PATH / 'chat_template.jinja'
    if chat_template.exists():
        cmd += ['--chat-template', str(chat_template)]

    return cmd


def configure_cache_diagnostics(manifest: dict, server_env: dict, cmd: list[str]) -> None:
    '''Install observational hooks only when the notebook explicitly selects test mode.'''
    namespace = {'__name__': 'arc3_cache_diagnostics'}
    exec(CACHE_DIAGNOSTICS_SOURCE, namespace)
    enabled = namespace['is_test_run']()
    server_env['ARC3_CACHE_DIAGNOSTICS'] = '1' if enabled else '0'
    server_env.pop('ARC3_CACHE_DIAGNOSTICS_DIR', None)
    if not enabled:
        return
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    (dist_packages / 'arc3_cache_diagnostics.py').write_text(CACHE_DIAGNOSTICS_SOURCE, encoding='utf-8')
    metadata = dist_packages / 'arc3_cache_diagnostics-1.0.dist-info'
    metadata.mkdir(exist_ok=True)
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: arc3-cache-diagnostics\nVersion: 1.0\n')
    (metadata / 'entry_points.txt').write_text(
        '[vllm.general_plugins]\narc3_cache_diagnostics = arc3_cache_diagnostics:install_cache_diagnostics\n'
    )
    if 'VLLM_PLUGINS' in server_env:
        server_env['VLLM_PLUGINS'] += ',arc3_cache_diagnostics'
    server_env['ARC3_CACHE_DIAGNOSTICS_DIR'] = str(WORKING_DIR / 'metrics')
    cmd += ['--kv-cache-metrics', '--kv-cache-metrics-sample', '0.1']


def should_enable_vllm_profiler() -> bool:
    '''Fail closed: a request is honored only in an explicit non-competition test run.'''

    return VLLM_PROFILE_REQUESTED and is_test_run()


def configure_vllm_profiler(cmd: list[str]) -> None:
    '''Add the bounded worker profiler without touching production commands.'''
    if not should_enable_vllm_profiler():
        return
    VLLM_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    cmd += [
        '--profiler-config',
        json.dumps({
            'profiler': 'torch',
            'torch_profiler_dir': str(VLLM_PROFILE_DIR),
            'torch_profiler_with_stack': False,
            'torch_profiler_use_gzip': True,
            'torch_profiler_dump_cuda_time_total': True,
            'torch_profiler_record_shapes': True,
            'torch_profiler_with_memory': False,
            'ignore_frontend': True,
            # The pinned WorkerProfiler starts on count == delay before
            # execute_model, so 6 skips exactly 5 iterations. It also stops on
            # profiled_count > max before execute_model, making 20 the number
            # of model iterations captured despite the internal counter reaching 21.
            'delay_iterations': 6,
            'max_iterations': 20,
        }),
    ]


def start_vllm_profiler() -> None:
    '''Start profiling after startup and smoke traffic, immediately before ARC requests.'''
    if not should_enable_vllm_profiler():
        return
    request_no_content(f'{VLLM_SERVER_URL}/start_profile', timeout=30)
    print(
        'vLLM profiler armed: skip 5 engine iterations, capture 20; '
        f'traces will be written to {VLLM_PROFILE_DIR}',
        flush=True,
    )


def build_watchdog_config(cmd: list[str]) -> dict:
    '''Configure the server watchdog and test-only host RAM sampling.'''
    watchdog_config = {
        'command': cmd,
        'server_log': str(VLLM_SERVER_LOG),
        'server_pid': str(VLLM_SERVER_PID),
        'server_ready': str(VLLM_SERVER_READY),
        'model_path': str(MODEL_PATH),
        'shard_prefetch_report': str(VLLM_SHARD_PREFETCH_REPORT),
    }
    if is_test_run():
        watchdog_config['host_memory_log'] = str(WORKING_DIR / 'metrics' / 'host_memory.jsonl')

    return watchdog_config


def start_vllm_server() -> None:
    '''Unpack the image runtime and launch the vLLM OpenAI server against the mounted model.

    The server is not spawned directly but through a watchdog process that becomes its parent.
    A CUDA illegal memory access poisons the engine, and the API server process exits in the same
    second, so the watchdog notices through a blocking wait on its child with no polling at all. It
    logs the exit and the tail of the server log to VLLM_WATCHDOG_LOG and starts the server again.
    The harness retries a failed request every second until its deadline, so games resume on their
    own once the new server binds the port. A server that dies before VLLM_SERVER_READY exists is
    not restarted, so a broken launch still fails this setup step quickly, and the watchdog gives up
    after a fixed number of restarts so a deterministic crash does not loop for the rest of the run.

    A separate process prefetches the GPU shards a bounded window ahead of each weight load
    (GpuShardPrefetcher), including restarts. It follows only the current start's server log and
    finishes long before the server is ready. If the server dies first, the watchdog terminates
    prefetch with bounded waits before restarting, so a blocked NFS read cannot delay recovery
    indefinitely.
    '''
    manifest = install_image_runtime()
    server_env = build_image_runtime_env(manifest)
    cmd = build_vllm_server_command()
    configure_cache_diagnostics(manifest, server_env, cmd)
    configure_vllm_profiler(cmd)
    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    VLLM_SERVER_LOG.unlink(missing_ok=True)
    VLLM_SHARD_PREFETCH_REPORT.unlink(missing_ok=True)
    VLLM_SERVER_PID.unlink(missing_ok=True)
    VLLM_SERVER_READY.unlink(missing_ok=True)
    if os.environ.get('TAAF_RUN_AS_SUBMISSION') == '1':
        cmd += ['--disable-log-stats', '--disable-uvicorn-access-log']
    else:
        cmd += ['--enable-prompt-tokens-details']

    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)
    VLLM_WATCHDOG_SCRIPT.write_text(WATCHDOG_SCRIPT_TEXT, encoding='utf-8')
    VLLM_WATCHDOG_CONFIG.write_text(json.dumps(build_watchdog_config(cmd)), encoding='utf-8')
    watchdog_log_handle = VLLM_WATCHDOG_LOG.open('w', encoding='utf-8')
    watchdog_process = subprocess.Popen(
        [sys.executable, str(VLLM_WATCHDOG_SCRIPT), str(VLLM_WATCHDOG_CONFIG)],
        env=server_env,
        stdout=watchdog_log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    VLLM_WATCHDOG_PID.write_text(str(watchdog_process.pid), encoding='utf-8')
    wait_for_vllm_server(watchdog_process, VLLM_SERVER_READY_TIMEOUT_SECONDS)
    VLLM_SERVER_READY.write_text(datetime.now().isoformat(timespec='seconds'), encoding='utf-8')


def print_vllm_backend_summary() -> None:
    markers = (
        'attention backend',
        'attentionbackendenum',
        'kv cache layout',
        'kernel for',
        'block fp8',
        'prefill kernel',
        'gpu kv cache size',
        'maximum concurrency',
    )
    log_lines = VLLM_SERVER_LOG.read_text(encoding='utf-8', errors='replace').splitlines()
    print('\n' + '=' * 88, flush=True)
    print('VLLM ATTENTION BACKEND / KV CACHE SUMMARY', flush=True)
    for line in log_lines:
        if any(marker in line.lower() for marker in markers):
            print(line, flush=True)
    print('=' * 88 + '\n', flush=True)


def run_vllm_api_smoke_test() -> None:
    payload = {
        'model': SERVED_MODEL_NAME,
        'messages': [{'role': 'user', 'content': 'Answer in one short sentence: what is 2 + 2?'}],
        'temperature': 0.0,
        'max_tokens': 96,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    response = request_json(f'{VLLM_BASE_URL}/chat/completions', payload=payload, timeout=120)
    generated = response['choices'][0]['message'].get('content', '').strip()
    print('\n' + '=' * 88, flush=True)
    print('VLLM OPENAI SERVER QWEN SMOKE TEST REAL MODEL OUTPUT', flush=True)
    print('Generated:', generated, flush=True)
    print('=' * 88 + '\n', flush=True)


print(f'vLLM runtime dataset path: {RUNTIME_DATASET}', flush=True)
print(f'Qwen model path: {MODEL_PATH}', flush=True)
assert_expected_cuda_gpu()
missing = [str(path) for path in (RUNTIME_DATASET, MODEL_PATH) if not path.exists()]
if missing:
    raise FileNotFoundError(
        'Missing attached input path(s): ' + ', '.join(missing) + '\n' + describe_kaggle_input_tree()
    )
start_vllm_server()
print_vllm_backend_summary()
if os.environ.get('TAAF_RUN_AS_SUBMISSION') == '1':
    print('Skipping vLLM smoke test (submission mode)', flush=True)
else:
    run_vllm_api_smoke_test()
start_vllm_profiler()
setup_env = {
    'ARC3_CACHE_DIAGNOSTICS': '1' if is_test_run() else '0',
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'LOCAL_ANALYZER_BASE_URL': VLLM_BASE_URL,
    'OPENAI_BASE_URL': VLLM_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'OPENAI_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW),
    'LOCAL_ANALYZER_TARGET_CONTEXT': '81536',
    'LOCAL_ANALYZER_MAX_OUTPUT': '0',
    'LOCAL_ANALYZER_TOOL_STEPS': '0',
    'LOCAL_ANALYZER_TOOL_TIMEOUT': '30',
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '1024',
    'LOCAL_ANALYZER_YIELD_SECONDS': '150',
    'LOCAL_ANALYZER_TEMPERATURE': '0.6',
    'LOCAL_ANALYZER_TOP_P': '0.95',
    'LOCAL_ANALYZER_TOP_K': '20',
    'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
    'LOCAL_ANALYZER_REASONING_EFFORT': 'xhigh',
    'MULTIMODAL_CONTEXT': 'current_grid',
    'MULTIMODAL_UPSCALE': '4',
}
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
existing_setup_env = {}
if setup_env_path.exists():
    existing_setup_env = json.loads(setup_env_path.read_text(encoding='utf-8'))
    if not isinstance(existing_setup_env, dict):
        raise RuntimeError('TAAF_KAGGLE_SETUP_ENV must contain a JSON object.')
existing_setup_env.update(setup_env)
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')


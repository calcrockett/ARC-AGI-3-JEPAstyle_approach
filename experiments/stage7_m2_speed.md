# Stage 7 — serving speed on the incumbent (milestone-2 fork + level memory)

**Why.** The run is decode-bound and short on tokens (~200K/game vs 27-67K
per level). Our check run: decode plateaus ~700-760 tok/s at 8-10 running
requests, KV pool peaks at 91%, 4.25 GB GPU memory left free, speculative
accept length ~2.7 of 4. The other two Milestone-2 winners peak at 1,135 and
1,159 tok/s.

**Variants** (`scripts/_build_m2_speed_kernels.py`; serving knobs only; every
variant plays all 25 public games for 25 min so more games compete than there
are streams, as in the hidden run; never submitted):

| variant | change |
|---|---|
| base | incumbent serving: 3 draft steps, mem 0.96, 10 streams |
| spec4 | 4 draft steps (lossless) |
| s12 | mem 0.98, 12 streams |
| s14 | mem 0.98, 14 streams, Mamba cache 72 |
| s14hic | s14 + 32 GB hierarchical KV cache in system RAM (~64 GB free of 177) |

Kaggle allows 2 GPU sessions: base and spec4 ran together (fair pair); the
rest are pushed by a detached queue (`scripts/kaggle_push_queue.py`, task
`ARC3PushQueue`). Report: `scripts/m2_speed_report.py` -> `logs/m2_speed_report.md`.

**Pre-registration.** Primary metric: generated tokens/s over the run (from
the harness summary), with decode tok/s by concurrent requests from the server
log as the mechanism. A variant is adopted only if it beats base on generated
tokens/s by > 5% with no tracebacks, no retraction storm (retracts within 2x
base) and prefix reuse >= 90%. spec4 is lossless, so it may ship on speed alone;
stream/memory variants change cache pressure and are judged on all four.
s14hic must show `hicache_attached=True`, else it is a no-op, not a result.
Public-game scores are reported but cannot rank (one 25-game run each).

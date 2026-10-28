# SPDX-License-Identifier: Apache-2.0
"""Environment configuration of the remote DSpark proxy (Spark side).

The overlay is inactive unless VLLM_DSPARK_REMOTE is set (then the image's own code
path runs byte-identically). Every value is validated at startup; a bad value raises.

  VLLM_DSPARK_REMOTE              endpoint for rank 0: o1rdma://10.202.28.1:18620 (lane O1's ofl, RDMA),
                                  o1tcp://... (ofl's TCP backend) or tcp://... (the contract's TCP stub)
  VLLM_DSPARK_REMOTE_BIND         this side's direct-link IPv4 (10.202.28.2); same as ?bind= in the
                                  endpoint, without a '?' that a launcher shell could glob
  VLLM_DSPARK_REMOTE_OFL_DIR      directory holding libofl.so + ofl.py (o1tcp/o1rdma endpoints)
  VLLM_DSPARK_REMOTE_POLICY       shadow (default) | resilience | capacity
  VLLM_DSPARK_REMOTE_IDENTITY     64-hex drafter identity (CONTRACT 6.7); required
  VLLM_DSPARK_REMOTE_MARKOV_IDENTITY  64-hex; default: sha256 of the VLLM_DSPARK_MARKOV_CACHE
                                  file, zeros if that variable is unset
  VLLM_DSPARK_REMOTE_TIMEOUT_MS   per-round deadline (default 50)
  VLLM_DSPARK_REMOTE_CONNECT_TIMEOUT_MS  connect + HELLO deadline (default 1000)
  VLLM_DSPARK_REMOTE_BACKOFF_MS   first reconnect delay after a failure (default 1000,
                                  doubles up to VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS, default 30000)
  VLLM_DSPARK_REMOTE_ON_FAIL      target_only (default) | raise   (capacity policy)
  VLLM_DSPARK_REMOTE_TRIP         circuit breaker (CONTRACT 10, breaker.py): consecutive failed attempts
                                  before the remote is skipped (default 3; 0 = off)
  VLLM_DSPARK_REMOTE_COOLDOWN     rounds without network after a trip, then one probe (default 256)
  VLLM_DSPARK_REMOTE_STRICT_START 1: raise at startup if connect + HELLO fails (default 0)
  VLLM_DSPARK_REMOTE_SHADOW_USE   local (default) | remote        (shadow policy)
  VLLM_DSPARK_REMOTE_WANT_MARGIN  1: ask the worker for top1-top2 margins (default 1 in shadow)
  VLLM_DSPARK_REMOTE_DRAFT_MS     remote draft round cost for the depth policy (ms, finite >= 0);
                                  replaces VLLM_DSPARK_DRAFT_MS in capacity/resilience
  VLLM_DSPARK_REMOTE_ROUND_MS     verify-only forward table for remote mode: ms for depth 0..k
                                  (k + 1 finite values > 0, non-decreasing); replaces
                                  VLLM_DSPARK_ROUND_MS (whose live value includes the local
                                  drafter) in capacity/resilience. With VLLM_DSPARK_DEPTH_POLICY
                                  on, capacity/resilience need BOTH remote costs (no mixed cost
                                  model) and refuse "confr:<table>" (its own table includes the
                                  local drafter). Shadow keeps the local measurements.
  VLLM_DSPARK_REMOTE_ASYNC        1: capacity policy only - rank 0 does not block its CPU on the round;
                                  the engine stream and the round's proxy thread hand each round over
                                  through two words in registered host memory (streamsync.HostFlagSync);
                                  the proxy thread makes no CUDA call. Self-tested at startup; falls back
                                  to the synchronous round with a warning. Default 0.
  VLLM_DSPARK_REMOTE_ASYNC_PROXY  async mode: where the round's proxy runs. thread (default): a Python thread of the
                                  engine process (needs the engine's GIL to publish). process: a torch-free child
                                  process with its own interpreter (proxy_proc.py); the gather staging, the descriptor
                                  ring and the hand-off words share one registered memfd region (shmregion.py), so the
                                  success path never waits for the engine's GIL; the switch interval is not changed
  VLLM_DSPARK_REMOTE_HOSTFLAG_LIB async mode: path of libdsv41_hostflag.so (built from
                                  remote_dspark/hostflag.c). Default: VLLM_DSPARK_REMOTE_OFL_DIR's copy,
                                  else hostflag.c compiled with cc at startup.
  VLLM_DSPARK_REMOTE_ASYNC_NAP_US async mode: the proxy thread spins on the gathered word; > 0 = sleep this
                                  many us between checks after VLLM_DSPARK_REMOTE_ASYNC_SPIN_US of
                                  spinning (default 0: spin; one rank-0 core while a round is pending)
  VLLM_DSPARK_REMOTE_ASYNC_SPIN_US see ASYNC_NAP_US (default 1000)
  VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS async mode: a native watchdog fails and releases a round the engine stream
                                  gathered but nobody released within this many ms (zero drafts for that round), so a
                                  proxy thread that cannot run never parks the GPU for good. Default 0 = auto:
                                  max(4 x VLLM_DSPARK_REMOTE_TIMEOUT_MS, 200)
  VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS  async process mode: new proxy processes the engine may start after its proxy died
                                  or stalled (default 3; 0 = none: target-only rounds for the rest of the run). Between
                                  the loss and the new proxy's start every round is target-only (zero drafts); the new
                                  proxy starts with a new session, so the worker drops its windows and every slot is
                                  re-announced (NEW_GEN). Spent: target-only rounds, one error log line
  VLLM_DSPARK_REMOTE_ASYNC_RESPAWN_RESET_S  a proxy process that served this long restores the full respawn budget
                                  (default 3600; 0 = never)
  VLLM_DSPARK_REMOTE_SWITCH_INTERVAL_US  async thread mode: Python GIL switch interval while the proxy thread
                                  runs (default 500; the interpreter default is 5000; 0 = unchanged); ignored with
                                  ASYNC_PROXY=process
  VLLM_DSPARK_REMOTE_FUSED_GATHER auto (default: on CUDA, after a startup self-check against the torch
                                  path) | 0 | 1: the one-launch Triton row gather (fused_gather.py)
  VLLM_DSPARK_REMOTE_N_OUT        auto (default) | 0: ReqEntry.n_out (CONTRACT 6.3). auto = cap[n] of the dynamic
                                  speculation schedule when the depth policy masks positions >= cap
                                  (conf/confb/confr, no VLLM_DSPARK_DEPTH_FILE) in capacity/resilience;
                                  otherwise, and in shadow, always all N positions
  VLLM_DSPARK_REMOTE_LOG_EVERY    rounds between stats log lines (default 1024, 0 = off)
  VLLM_DSPARK_REMOTE_STATS_FILE   JSON stats written by rank 0 (default
                                  /tmp/dsv41-remote-dspark.json, inside the container)
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

from . import wire
from .transport import Endpoint, parse_endpoint

POLICIES = ("shadow", "resilience", "capacity")
ENV_ENDPOINT = "VLLM_DSPARK_REMOTE"


def remote_requested(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return bool((env.get(ENV_ENDPOINT) or "").strip())


def _int(env, key: str, default: int, lo: int = 0) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
    except ValueError as e:
        raise ValueError(f"{key}={raw!r}: not an integer") from e
    if v < lo:
        raise ValueError(f"{key}={v}: must be >= {lo}")
    return v


def _choice(env, key: str, default: str, choices: tuple[str, ...]) -> str:
    v = (env.get(key) or default).strip().lower()
    if v not in choices:
        raise ValueError(f"{key}={v!r}: expected one of {choices}")
    return v


def _bool(env, key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "on", "yes"):
        return True
    if v in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"{key}={raw!r}: expected 0 or 1")


def _float_list(env, key: str) -> list[float] | None:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return None
    try:
        vals = [float(x) for x in raw.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"{key}={raw!r}: expected comma-separated numbers") from e
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{key}={raw!r}: expected finite numbers")
    if any(v < 0 for v in vals):
        raise ValueError(f"{key}={raw!r}: expected non-negative numbers")
    return vals


@dataclass
class RemoteConfig:
    endpoint: Endpoint
    policy: str
    drafter_identity: bytes
    markov_identity: bytes
    timeout_ms: int
    connect_timeout_ms: int
    backoff_ms: int
    backoff_max_ms: int
    on_fail: str
    strict_start: bool
    shadow_use: str
    want_margin: bool
    remote_draft_ms: float | None
    remote_round_ms: list[float] | None
    log_every: int
    stats_file: str
    async_mode: bool = False
    switch_interval_us: int = 500
    hostflag_lib: str = ""
    async_nap_us: int = 0
    async_spin_us: int = 1000
    async_watchdog_ms: int = 0
    async_proxy: str = "thread"
    async_respawns: int = 3
    async_respawn_reset_s: int = 3600
    fused_gather: bool | None = None
    n_out: str = "auto"
    trip: int = 3
    cooldown: int = 256

    @property
    def watchdog_ms(self) -> int:
        """The async watchdog deadline actually used."""
        return self.async_watchdog_ms or max(4 * self.timeout_ms, 200)

    @property
    def keeps_local_drafter(self) -> bool:
        return self.policy in ("shadow", "resilience")

    @classmethod
    def from_env(cls, environ=None) -> "RemoteConfig":
        env = os.environ if environ is None else environ
        if not remote_requested(env):
            raise ValueError(f"{ENV_ENDPOINT} is not set")
        endpoint = parse_endpoint(env[ENV_ENDPOINT])
        bind = (env.get("VLLM_DSPARK_REMOTE_BIND") or "").strip()
        if bind:
            if endpoint.options.get("bind", bind) != bind:
                raise ValueError(f"VLLM_DSPARK_REMOTE_BIND={bind} conflicts with the endpoint's "
                                 f"?bind={endpoint.options['bind']}")
            endpoint.options["bind"] = bind
        policy = _choice(env, "VLLM_DSPARK_REMOTE_POLICY", "shadow", POLICIES)
        ident_raw = env.get("VLLM_DSPARK_REMOTE_IDENTITY", "")
        if not ident_raw.strip():
            raise ValueError(
                "VLLM_DSPARK_REMOTE_IDENTITY is required: the 64-hex drafter identity "
                "(offload/dspark/spark/tools/identity.py over lane S2's DSPARK_COPY_DONE.json)")
        drafter_identity = wire.parse_hex32(ident_raw, "VLLM_DSPARK_REMOTE_IDENTITY")
        markov_raw = env.get("VLLM_DSPARK_REMOTE_MARKOV_IDENTITY", "")
        if markov_raw.strip():
            markov_identity = wire.parse_hex32(markov_raw, "VLLM_DSPARK_REMOTE_MARKOV_IDENTITY")
        else:
            markov_identity = markov_identity_from_env(env)
        draft_ms_list = _float_list(env, "VLLM_DSPARK_REMOTE_DRAFT_MS")
        if draft_ms_list is not None and len(draft_ms_list) != 1:
            raise ValueError("VLLM_DSPARK_REMOTE_DRAFT_MS: one number (ms)")
        cfg = cls(
            endpoint=endpoint,
            policy=policy,
            drafter_identity=drafter_identity,
            markov_identity=markov_identity,
            timeout_ms=_int(env, "VLLM_DSPARK_REMOTE_TIMEOUT_MS", 50, lo=1),
            connect_timeout_ms=_int(env, "VLLM_DSPARK_REMOTE_CONNECT_TIMEOUT_MS", 1000, lo=1),
            backoff_ms=_int(env, "VLLM_DSPARK_REMOTE_BACKOFF_MS", 1000, lo=1),
            backoff_max_ms=_int(env, "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS", 30000, lo=1),
            on_fail=_choice(env, "VLLM_DSPARK_REMOTE_ON_FAIL", "target_only",
                            ("target_only", "raise")),
            strict_start=_bool(env, "VLLM_DSPARK_REMOTE_STRICT_START", False),
            shadow_use=_choice(env, "VLLM_DSPARK_REMOTE_SHADOW_USE", "local",
                               ("local", "remote")),
            want_margin=_bool(env, "VLLM_DSPARK_REMOTE_WANT_MARGIN", policy == "shadow"),
            remote_draft_ms=None if draft_ms_list is None else draft_ms_list[0],
            remote_round_ms=_float_list(env, "VLLM_DSPARK_REMOTE_ROUND_MS"),
            log_every=_int(env, "VLLM_DSPARK_REMOTE_LOG_EVERY", 1024, lo=0),
            stats_file=(env.get("VLLM_DSPARK_REMOTE_STATS_FILE")
                        or "/tmp/dsv41-remote-dspark.json"),
            async_mode=_bool(env, "VLLM_DSPARK_REMOTE_ASYNC", False),
            switch_interval_us=_int(env, "VLLM_DSPARK_REMOTE_SWITCH_INTERVAL_US", 500, lo=0),
            hostflag_lib=(env.get("VLLM_DSPARK_REMOTE_HOSTFLAG_LIB") or "").strip(),
            async_nap_us=_int(env, "VLLM_DSPARK_REMOTE_ASYNC_NAP_US", 0, lo=0),
            async_spin_us=_int(env, "VLLM_DSPARK_REMOTE_ASYNC_SPIN_US", 1000, lo=0),
            async_watchdog_ms=_int(env, "VLLM_DSPARK_REMOTE_ASYNC_WATCHDOG_MS", 0, lo=0),
            async_proxy=_choice(env, "VLLM_DSPARK_REMOTE_ASYNC_PROXY", "thread", ("thread", "process")),
            async_respawns=_int(env, "VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS", 3, lo=0),
            async_respawn_reset_s=_int(env, "VLLM_DSPARK_REMOTE_ASYNC_RESPAWN_RESET_S", 3600, lo=0),
            fused_gather={"auto": None, "1": True, "0": False}[
                _choice(env, "VLLM_DSPARK_REMOTE_FUSED_GATHER", "auto", ("auto", "0", "1"))],
            n_out=_choice(env, "VLLM_DSPARK_REMOTE_N_OUT", "auto", ("auto", "0")),
            trip=_int(env, "VLLM_DSPARK_REMOTE_TRIP", 3, lo=0),
            cooldown=_int(env, "VLLM_DSPARK_REMOTE_COOLDOWN", 256, lo=1),
        )
        if cfg.async_mode and cfg.policy != "capacity":
            raise ValueError("VLLM_DSPARK_REMOTE_ASYNC=1 needs VLLM_DSPARK_REMOTE_POLICY=capacity "
                             "(resilience/shadow decide on the host every round)")
        if cfg.async_mode and cfg.on_fail == "raise":
            raise ValueError("VLLM_DSPARK_REMOTE_ASYNC=1 cannot raise on a failed round "
                             "(the host never waits for it); use ON_FAIL=target_only")
        if cfg.backoff_max_ms < cfg.backoff_ms:
            raise ValueError("VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS < VLLM_DSPARK_REMOTE_BACKOFF_MS")
        return cfg


def markov_identity_from_env(env) -> bytes:
    path = (env.get("VLLM_DSPARK_MARKOV_CACHE") or "").strip()
    if not path:
        return bytes(32)
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).digest()
    except OSError as e:
        raise ValueError(
            f"VLLM_DSPARK_MARKOV_CACHE={path!r} unreadable ({e}); set "
            "VLLM_DSPARK_REMOTE_MARKOV_IDENTITY explicitly") from e


def apply_cost_overrides(cfg: RemoteConfig, environ=None, depth_policy_on: bool = False,
                         depth_mode: str | None = None, num_spec: int = 5,
                         depth_file: str | None = None) -> dict[str, str]:
    """Make the remote round cost an explicit input of the depth policy (CONTRACT 10,
    brief "recalibration"). Writes VLLM_DSPARK_DRAFT_MS / VLLM_DSPARK_ROUND_MS in the
    process environment before the model runner builds DSparkDepthPolicy (it reads them
    in initialize_kv_cache, after the speculator exists; it is built only when
    VLLM_DSPARK_DEPTH_POLICY is on). Same env on both ranks, so the cost tables stay
    identical across TP ranks. Returns what was changed; raises before writing anything.

    The policy prices depth d as ROUND_MS[d] + DRAFT_MS (depth_policy.round_cost_ms). The
    live ROUND_MS includes the local drafter, so with the policy on, capacity/resilience
    need both remote inputs: a remote DRAFT_MS over the live table counts the drafter
    twice, a verify-only table under the live DRAFT_MS counts the wrong drafter. confr's
    table is part of the policy string and cannot take them at all. fixed:<d> never prices
    a depth, so it needs them only when a depth file can switch the mode at runtime."""
    env = os.environ if environ is None else environ
    changed: dict[str, str] = {}
    if cfg.policy == "shadow":
        return changed  # local drafter runs every round; timing is not representative
    if depth_policy_on and not (depth_mode == "fixed" and not depth_file):
        missing = [k for k, v in (("VLLM_DSPARK_REMOTE_DRAFT_MS", cfg.remote_draft_ms),
                                  ("VLLM_DSPARK_REMOTE_ROUND_MS", cfg.remote_round_ms))
                   if v is None]
        if missing:
            raise ValueError(
                f"remote DSpark ({cfg.policy}) with VLLM_DSPARK_DEPTH_POLICY needs both "
                "VLLM_DSPARK_REMOTE_DRAFT_MS (measured remote round, ms) and "
                "VLLM_DSPARK_REMOTE_ROUND_MS (verify-only table, ms by depth 0..k): missing "
                + " and ".join(missing) + ". The live VLLM_DSPARK_ROUND_MS includes the local "
                "drafter, so a partial override mixes two cost models")
        if depth_mode == "confr":
            raise ValueError(
                f"remote DSpark ({cfg.policy}): VLLM_DSPARK_DEPTH_POLICY=confr:<rows ms> carries "
                "its own step table (measured with the local drafter), which the remote cost "
                "inputs cannot replace; use conf or confb with VLLM_DSPARK_REMOTE_DRAFT_MS / "
                "VLLM_DSPARK_REMOTE_ROUND_MS")
    if cfg.remote_round_ms is not None:
        t = cfg.remote_round_ms
        if len(t) != num_spec + 1:
            raise ValueError(f"VLLM_DSPARK_REMOTE_ROUND_MS: {len(t)} values, expected "
                             f"{num_spec + 1} (verify ms for depth 0..{num_spec})")
        if any(v <= 0 for v in t):
            raise ValueError("VLLM_DSPARK_REMOTE_ROUND_MS: verify times must be > 0 ms")
        if any(b < a for a, b in zip(t, t[1:])):
            raise ValueError("VLLM_DSPARK_REMOTE_ROUND_MS: must be non-decreasing with the "
                             f"depth (more live rows never verify faster): {t}")
        changed["VLLM_DSPARK_ROUND_MS"] = ",".join(f"{v:g}" for v in t)
    if cfg.remote_draft_ms is not None:
        changed["VLLM_DSPARK_DRAFT_MS"] = f"{cfg.remote_draft_ms:g}"
    for k, v in changed.items():
        env[k] = v
    return changed


# depth-policy modes that zero the confidences at positions >= cap[num_reqs] before choosing
# depths (DSparkDepthPolicy.record); "fixed" and "off" verify by the scheduler window only
CAP_MASKING_DEPTH_MODES = ("conf", "confb", "confr")


def n_out_table(cfg: RemoteConfig, num_spec: int, depth_mode: str, depth_file: str | None,
                dense_cap: list[int] | None) -> tuple[list[int] | None, str]:
    """ReqEntry.n_out by request count (CONTRACT 6.3), or None (always all N positions),
    with the reason. dense_cap is build_dynamic_sd_schedule_lookup's table (index = number
    of requests, index 0 unused), as the model runner passes it to DSparkDepthPolicy.

    Positions >= cap[n] are only safe to drop when nothing reads them: the depth policy
    must zero them (a cap-masking mode, fixed for the life of the serve, so no depth file
    that could switch it to "fixed"/"off"), and shadow compares every position."""
    if cfg.policy == "shadow":
        return None, "shadow compares every position"
    if cfg.n_out == "0":
        return None, "VLLM_DSPARK_REMOTE_N_OUT=0"
    if depth_mode not in CAP_MASKING_DEPTH_MODES:
        return None, f"depth policy {depth_mode!r} does not mask positions >= cap"
    if depth_file:
        return None, "VLLM_DSPARK_DEPTH_FILE can switch the depth policy at runtime"
    if not dense_cap:
        return None, "no dynamic speculation schedule"
    table = [0 if i == 0 else min(num_spec, int(c)) for i, c in enumerate(dense_cap)]
    if all(c >= num_spec for c in table[1:]):
        return None, "schedule never caps below N"
    return table, "cap by request count"


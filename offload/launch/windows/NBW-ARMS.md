# NBW arms — fixed before measurement

CPU preparation only. The coordinator alone may run the attended window. Production
is 596ba1cf4e3b11be29b41a57e7f69a5d8efa4799,
tag `prod-k1-327-ops1-20261026`: K1 327680, KV bytes 6442450944,
engine pool 2502600, four sequences, TP2 K2 worker fb68c5291792ab90c0f468bd94010917967799f3,
capacity policy, async process proxy. Both Sparks must report kernel
7.0.0-1019-nvidia and driver 580.178.04. These are repository/brief facts,
not a new live inventory.

`NBW-native.env` reads production's own three OFFLOAD settings in a subshell,
preserving its read-only mounts, Engram REQUIRED setting, timing prices, request
caps, target model, K1 and async settings. It never mounts this checkout's engine
code. The common experimental changes are port 8001 and the run label.

| Arm | Worker on the same NVLink pair | Spark env changes from production |
|---|---|---|
| K | Enabled production TP2 K2 unit, unchanged current link | None |
| I | Transient native-hybrid-kx2 from reviewed NB1 source, invariant level 2, N4 prebuilt binaries | Exactly VLLM_DSPARK_REMOTE_IDENTITY and VLLM_DSPARK_REMOTE_MARKOV_IDENTITY |

I runs on GPU-c4e7a371-d8a8-ebf0-918c-da3e38c87a93 and
GPU-7744b44b-8a13-b06d-a67e-f958fd74b8fc under both nonblocking locks,
`prod_window.sh pair COORD`, same TP2/1048576 HELLO bound/listener.
No production checkout, current link, enabled unit, image or target weights change.
NB1's scratch source was deleted at closeout: the coordinator must stage a durable,
clean NB1 tree (item 2 dc6de03 or a reviewed descendant containing that change),
reuse N4's lib/build/p2p/caches and perform the solo identity prerequisite before HOLD.
No download or build is part of NBW.

The native digest hashes runtime sources and the invariant recipe. W8's 5c3be5c7
digest is invalid for this arm. The runner requires full source commit, durable
relative source-tree path, and both reviewed identity digests from the solo receipt.
It checks the running worker's status and recomputes the native identity from
that source's verified checkpoint receipt with level 2. Missing or different
identity means STOP; never discover and silently accept a new digest in HOLD.

Pre-registered rule: per cell, take the median of each block's per-request decode
tok/s, then the median across blocks for each arm. K-to-K spread is max(K block
medians) minus min(K block medians), in tok/s, with no noise floor. I must exceed
K by strictly more than that spread at short B1 and 32K B1, and lose no more than
that spread at 32K B2 and B3. Require two complete starts per arm, matching public
prompt hashes and budgets, zero failed drafting rounds, and six identical greedy
checks at every I start. Otherwise NOT A CANDIDATE. This brief's strict greedy
STOP takes precedence over W8's historical near-tie allowance.

No serving measurements exist for NBW yet. N4/N5 fidelity qualifications remain.

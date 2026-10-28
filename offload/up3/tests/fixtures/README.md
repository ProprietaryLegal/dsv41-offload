Round 6, 2026-10-27: production-serve-rank0.sh, projected production-host-shapes.json
and filtered production-logs-HOST.txt were captured with authorized read-only
Docker/sysfs reads on sparkhead/sparkworker-wired. No prompts, generated text or model
bytes are included. Logs intentionally omit all unrelated messages.

logger-templates.json contains literal logger templates extracted without import
from the cited source. RDMA/pin-init examples rendered from them are source-format
fixtures, not claims that production emitted these arm messages. See
../../HOST-PARSE-AUDIT.md for exact source provenance, missing fields and limits.
Synthetic PID/thread/census values in boundary tests remain explicitly CPU fakes.

The production log fixtures also include the later item-3 probe's observed
single-ring message bodies (logger prefix stripped by the real parser). Initial
snapshot messages and later initialization observations are distinguished in the
audit. Config.Env in the projected fixture retains only EXL3_RDMA_DEVICES (and
EXL3_CPU_PIN if present); its list-of-KEY=value shape matches the full real Env.

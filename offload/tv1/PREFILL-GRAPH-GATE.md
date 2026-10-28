# Item 2d — gate FAIL, no prefill CUDA graphs built

MEASURED source: SP1-W-20261026T191247Z/trace-budget-v3.json, source SHA256
and complete per-rank aggregates pinned in exploration-pins.json. Profiling affects
absolute time; this decision uses the already-saved traces, no new GPU window.

| Capture | Rank | Median chunk ms | Total idle share | Host gap share |
|---|---:|---:|---:|---:|
| 32K early | 0 | 1854.544 | 0.9825% | 0.4858% |
| 32K early | 1 | 1854.464 | 1.0345% | 0.6617% |
| 150K late | 0 | 1886.238 | 0.5955% | 0.0954% |
| 150K late | 1 | 1886.226 | 0.4696% | 0.0852% |

Binding TV1 threshold: build only if existing data show host gap / idle >=5%.
Maximum TOTAL idle is 1.0345%, so the gate fails even treating every idle interval as
host launch overhead. ESTIMATED impossible upper bound from deleting all idle:
~9–19 ms per chunk, <=1.04%; actual graph savings would be smaller. No switch,
graph capture allocations, graph metadata or production files were added. #43733's
live-context bridge and #43736's persistent bounded-replay tail were explored in
EXPLORE.md; their GB300 gains do not transfer to these compute-heavy chunks.

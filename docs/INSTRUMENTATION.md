# APEXA Instrumentation

APEXA records structured latency and throughput data for every LLM call and
every user query. Use this to answer questions like:

- What's the median wall-clock per Argo call for `gpt55` vs `claudeopus48`?
- How many LLM round-trips does a "calibrate then integrate" query typically take?
- Where's the time going — LLM latency or tool execution?
- What are the real TTFT (time-to-first-token) and TPOT (time-per-output-token)?
- How does APEXA compare to NVIDIA's DGX Spark agentic workload benchmarks?

Zero overhead when off. When on, never breaks a running agent — telemetry
writes are best-effort and swallow all errors.

---

## What each tier does

| Tier | What it captures | JSONL `endpoint` |
|---|---|---|
| **1** | One row per HTTP call to Argo `/chat/` — elapsed, tokens, status, retry, error | `argo-chat` |
| **2** | One row per user query wrapping the above — round-trip count, LLM time, tool time, wall-clock, agent | `query` |
| **3** | Streaming provider via Anthropic-native `/messages/` — real **TTFT** and **TPOT** | `argo-messages` |

All rows go to the same JSONL log. The analyzer script consumes all three.

---

## Turning it on

```bash
export APEXA_SHOW_TIMING=1
# optional: change where the log is written
export APEXA_TIMING_LOG=~/.apexa/timing.jsonl   # default
# optional: switch to streaming provider for Claude models (Tier 3)
export APEXA_PROVIDER=streaming
```

Run APEXA as normal (`./start_beamline_assistant.sh`, `./start_gradio_ui.sh`,
`python web_server.py`). Every `ArgoProvider.chat()` (Tier 1) or
`StreamingAnthropicProvider.chat()` (Tier 3) HTTP call appends one row, and
every user query through `APEXAClient.run_query()` appends a summary row on
exit (Tier 2).

Turn it off by unsetting `APEXA_SHOW_TIMING`. No restart of any external
service required.

---

## Row schemas

### `endpoint = "argo-chat"` (Tier 1)

```json
{
  "iso_ts":       "2026-07-15T21:39:19.426086+00:00",
  "ts":           1784151559.426099,
  "endpoint":     "argo-chat",
  "query_id":     "37e8cb18c452",
  "model":        "gpt55",
  "attempt":      0,
  "http_status":  200,
  "elapsed_s":    1.234,
  "prompt_tok":   12034,
  "response_tok": 287,
  "gen_tps":      232.7,
  "n_messages":   8,
  "n_tool_calls": 1,
  "temperature":  1,
  "empty":        false,
  "error":        ""
}
```

| Field | Meaning |
|---|---|
| `query_id` | Links this call back to the `endpoint="query"` row for its parent query (empty if the call happened outside a `query_scope`, e.g. warmup) |
| `attempt` | 0-indexed retry attempt — a retried call appears as multiple rows, one per attempt |
| `http_status` | HTTP status; `0` on timeout |
| `elapsed_s` | Wall-clock for this HTTP round-trip. Includes network + queue + prefill + generation |
| `prompt_tok` | Token count of the messages array **as sent** (system + history + latest turn) |
| `response_tok` | Token count of the parsed content string returned by the model |
| `gen_tps` | `response_tok / elapsed_s` — combined prefill+generation because `/chat/` doesn't stream. For real gen tok/s use Tier 3 (`argo-messages`) |
| `n_tool_calls` | How many tool calls the model produced this turn |
| `empty` | True when the model returned neither content nor tool calls — the "silent greeting" degenerate case |
| `error` | Empty on success. `retry_502`, `retry_503`, `retry_429`, `http_400`, `timeout`, `exception:*` on failure. Rows tagged `retry_*` are not counted into Tier-2 aggregation |

### `endpoint = "argo-messages"` (Tier 3)

Adds the streaming-only metrics on top of the Tier 1 schema:

```json
{
  "endpoint":     "argo-messages",
  "ttft_s":       0.42,     // time from POST to first content_block_delta with text
  "tpot_ms":      9.5,      // (elapsed_s - ttft_s) / (response_tok - 1) * 1000
  "gen_tps":      107.5,    // (response_tok - 1) / (elapsed_s - ttft_s) — real generation rate
  "prompt_tok":   18000,    // from message_start.usage.input_tokens (real, not estimate)
  "response_tok": 290,      // from message_delta.usage.output_tokens (real)
  ...
}
```

Values come from the Anthropic streaming events (`message_start.usage`,
`message_delta.usage`) — no estimate. Falls back to tiktoken only if the
gateway doesn't return usage events. `ttft_s` is `null` if no text was ever
emitted (e.g. tool-only response); `tpot_ms` is `null` when `response_tok <= 1`.

### `endpoint = "query"` (Tier 2)

```json
{
  "iso_ts":            "...",
  "endpoint":          "query",
  "query_id":          "37e8cb18c452",
  "agent":             "calibration",
  "query_chars":       57,
  "n_llm_calls":       3,
  "sum_llm_elapsed_s": 5.3,
  "n_tools":           2,
  "tool_elapsed_s":    2.1,
  "wall_clock_s":      7.65,
  "llm_prompt_tok":    42500,
  "llm_response_tok":  730
}
```

| Field | Meaning |
|---|---|
| `agent` | Which specialist the orchestrator routed to (`calibration`, `analysis`, `knowledge`, `visualization`, `motor`); empty if a fast-path or short-circuit skipped routing |
| `n_llm_calls` | Count of successful/terminal LLM round-trips (mid-loop retries not counted) |
| `sum_llm_elapsed_s` | Sum of `elapsed_s` across all LLM calls this query made |
| `n_tools` / `tool_elapsed_s` | Count and total time of MCP tool executions |
| `wall_clock_s` | Full user-perceived latency. `wall_clock_s - sum_llm_elapsed_s - tool_elapsed_s` is orchestrator/routing overhead |
| `llm_prompt_tok` / `llm_response_tok` | Cumulative tokens across all LLM calls |

**Grouping:** filter Tier-1/3 rows by `query_id` to recover the exact
per-call breakdown for a given query.

---

## Streaming provider (Tier 3) — details

The blocking `/chat/` endpoint cannot report TTFT/TPOT: Argo returns a
single JSON blob after the response is complete. Tier 3 uses the
Anthropic-native `/messages/` endpoint (`https://apps.inside.anl.gov/argoapi/v1/messages`,
per CLAUDE.md) with `stream: true`.

**Enabled only when** `APEXA_PROVIDER=streaming` **AND** the model starts with
`claude`. Non-Claude models under `streaming` fall back to Argo `/chat/` with
a one-line stderr warning.

**Auth:** `x-api-key: <ANL_USERNAME>` + `anthropic-version: 2023-06-01` (matches
what the `anthropic` Python SDK sends). No new dependency added — raw `httpx`
streaming.

**Constraints:**

- `system` messages are extracted from the payload and passed as the
  top-level `system` string (Anthropic native doesn't accept `role: system`
  in messages).
- Opus 4.7/4.8 drop `temperature` (matches Argo's `/chat/` behavior for
  those models).
- Tool calls stream as `content_block_start`/`input_json_delta`/`content_block_stop`
  and are reconstructed into APEXA's normal `ToolCall` objects — so all
  downstream code (specialist agents, orchestrator, tool executor) works
  unchanged.

**Provider selection lives in** `select_provider()` in [apexa_agents.py](../apexa_agents.py).

---

## Analyzing the log

The analyzer script consumes the JSONL and prints three tables — this is the
output you'd bring to a benchmarking discussion.

```bash
./scripts/analyze_timing.py                       # default log path
./scripts/analyze_timing.py path/to/log.jsonl
./scripts/analyze_timing.py --since 2026-07-15    # filter by date
./scripts/analyze_timing.py --model gpt55         # single model
./scripts/analyze_timing.py --json                # machine-readable
```

Sample output:

```
=== Per-model LLM call summary ==========================================

endpoint         model            n  ret%  p50_s  p95_s  in_tok  out_tok    tps  ttft_s  tpot_ms
------------------------------------------------------------------------------------------------
argo-chat        gpt55            3 25.0%   1.20   2.28   11567     240  152.8      —       —
argo-messages    claudeopus48     2  0.0%   2.90   3.08   18750     270  107.8   0.40     9.20

=== Per-agent user-query summary ========================================

agent            n  wall_p50  wall_p95  llm_p50  tool_p50  calls  tools   in_tok  out_tok
----------------------------------------------------------------------------------------
calibration      1      5.20      5.20     3.60      0.90    2.0    1.0    26200      600
analysis         1      7.90      7.90     5.80      1.40    2.0    1.0    37500      540

=== User-perceived wall-clock (all queries) =============================

  p50      5.20 s
  p90      7.36 s
  p95      7.63 s
  ...
```

The tables answer the questions from the top of this doc directly. The
per-model table maps 1:1 to the columns NVIDIA uses in the DGX Spark blog
(elapsed / prefill / generation / TTFT / TPOT), so you can put APEXA on the
same axes as their benchmark numbers.

---

## Quick jq queries

```bash
# how many queries per agent today?
jq -r 'select(.endpoint=="query") | .agent' ~/.apexa/timing.jsonl \
  | sort | uniq -c

# median wall-clock per model (from LLM rows)
jq -r 'select(.endpoint|startswith("argo-")) | "\(.model)\t\(.elapsed_s)"' \
  ~/.apexa/timing.jsonl \
  | sort -k1,1 -k2n | awk '{by[$1]=by[$1]" "$2} END {for(m in by) print m, by[m]}'

# any empty completions?
jq -r 'select(.empty==true) | "\(.iso_ts) \(.model) \(.n_messages)msgs"' \
  ~/.apexa/timing.jsonl

# retry rate for /chat/
jq -r 'select(.error|startswith("retry_")) | .error' ~/.apexa/timing.jsonl \
  | sort | uniq -c

# recover per-call breakdown for a specific query
QID=37e8cb18c452
jq --arg qid $QID 'select(.query_id==$qid)' ~/.apexa/timing.jsonl
```

---

## Overhead and privacy

**Overhead when off** (`APEXA_SHOW_TIMING` unset): one `os.environ.get` per
call — negligible.

**Overhead when on:** one tiktoken encode of the full messages array (~5–15 ms
for a 15K-token prompt) + one JSON serialise + one line append (<1 ms). Tier 2
adds a `contextvars` set/reset per query and a second JSONL line at exit
(<1 ms). Insignificant next to the multi-second LLM round-trip.

**Privacy:** rows never include your ANL username, the message contents, the
model's response text, tool arguments, or file paths. Numerics only, plus the
model name and specialist-agent name. Safe to share with collaborators, attach
to bug reports, or publish alongside benchmark results.

---

## Where the code lives

- [apexa_timing.py](../apexa_timing.py) — logger, token counter, `query_scope`, context propagation
- [apexa_agents.py](../apexa_agents.py) — `ArgoProvider.chat()` + `_timing_record()` (Tier 1), `StreamingAnthropicProvider` + `select_provider()` (Tier 3)
- [argo_mcp_client.py](../argo_mcp_client.py) — `run_query()` wraps `query_scope`, `execute_tool_call()` records `tool_elapsed_s`
- [scripts/analyze_timing.py](../scripts/analyze_timing.py) — the summary-table generator

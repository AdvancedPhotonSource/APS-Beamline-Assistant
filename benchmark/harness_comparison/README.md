# Harness comparison — same backend, same task, different driver

Measures how much of an agent's reliability belongs to the **harness** rather than
the **model**. Published agent comparisons report model names and vary the harness
silently; to our knowledge none holds the harness fixed while varying the model, or
the reverse.

That is possible here because **argo-proxy exposes one OpenAI-compatible endpoint
that several harnesses already speak natively** — Claude Code, OpenCode, Codex CLI,
Aider, Gemini CLI — so model, harness, and transport can be varied one at a time
against identical infrastructure.

Harnesses under test: **APEXA, Claude Code, OpenCode, Pi** (`pi.dev`). EAA dropped.

---

## The measurement that matters

Every agent benchmark asks *was the answer right?* On a strict execution environment
we can ask the sharper question: **was the answer right because something ran?**

Each task therefore carries three independent facts:

| fact | how it is established |
|---|---|
| `reported` | parsed from the harness's final text |
| `ground_truth` | a value that exists only in a file produced by actually running the analysis |
| `executed` | artifacts on disk with an mtime later than the run started |

Which lets us separate outcomes that all look identical in a transcript:

- **grounded** — reported matches ground truth *and* artifacts were produced
- **fabricated** — reported a plausible value with **no fresh artifacts**
- **guessed** — reported the *nominal* value inferable from context (e.g. a distance
  in the filename) rather than the *refined* value only a run can yield
- **honest-fail** — declined, or reported failure, having produced nothing
- **wrong** — ran, but the value disagrees

`guessed` is the discriminator this design is built around. A CeO2 calibration file
named `..._650mm_...` invites the answer "650000 µm". The refined distance differs.
A harness that reports the nominal number is not reading the analysis it claims to
have run, and no transcript-graded benchmark can tell the difference.

## Task design rules

1. **Multi-step.** The answer must come from a value produced by an earlier step,
   not from the prompt. Single-call tasks measure nothing about agency.
2. **Unguessable ground truth.** The correct value must not be inferable from the
   filename, the prompt, or domain priors. Where a plausible wrong answer exists in
   context, record it as `nominal` so `guessed` can be scored separately.
3. **Real tools on real data.** No mock pipeline. Failures should be the ones that
   occur at a beamline.
4. **Bounded.** Minutes per trial; the comparison needs repeats for error bars.

## Controlling the confound

Two conditions, run separately, because they answer different questions:

- **`shell`** — every harness gets only its native file/shell tools and drives the
  MIDAS CLI by hand. Tool surface is held constant, so the harness is the only
  variable. This is the clean measurement.
- **`native`** — each harness uses whatever tool surface it actually ships with
  (APEXA its 81 typed MCP tools; Claude Code and OpenCode their MCP support; Pi
  deliberately has none). This measures the deployed systems as they exist, and the
  difference between conditions is what the typed tool surface buys.

Reporting only `native` would confound harness with tool surface, which is the error
this whole comparison exists to avoid.

## Prerequisites

```bash
argo-proxy serve                       # backend, ANL network/VPN
export APEXA_LLM_BASE_URL=http://localhost:<port>/v1
export ANL_USERNAME=<user>
```

Harness installs (none are on PATH by default). **Install them one at a time** — npm fails a multi-package command atomically, so one bad name blocks the rest:

| harness | install | non-interactive invocation |
|---|---|---|
| Claude Code | `npm i -g @anthropic-ai/claude-code` | `claude -p "<prompt>"` |
| OpenCode | `npm i -g opencode-ai` | `opencode run "<prompt>"` |
| Pi | `npm i -g --ignore-scripts @earendil-works/pi-coding-agent` | `pi -p "<prompt>"` |
| APEXA | in-tree | driven in-process by the runner |

Each is pointed at argo-proxy through its own base-URL/API-key environment; the
runner sets these per harness so the backend is identical across arms.

## Running

```bash
python benchmark/harness_comparison/run_comparison.py --self-test        # offline
python benchmark/harness_comparison/run_comparison.py \
    --harnesses apexa claude-code opencode pi \
    --condition shell --trials 5
```

Rows land in `results/` as JSONL, one per trial, with the classification, the
reported value, the artifacts observed, turn count and wall-clock.

## Honest limits

Harness parity is approximate by construction: harnesses differ in context
management, retry policy and error surfacing *by design*, and equalising them
entirely would defeat the comparison. What is held identical is the backend, the
model, the task, the working directory, and — in the `shell` condition — the tool
surface. Everything else is the thing being measured.

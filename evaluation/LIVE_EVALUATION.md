# Evaluating the real Gemini model

Everything below is written from this sandbox, which has **no network path to Google's API** (confirmed directly:
`curl` to `generativelanguage.googleapis.com` fails outright here — only PyPI/npm/GitHub/Anthropic are reachable).
So none of this has actually been run against the live model. It has been run, and works, with `--model offline`
and every `mock-*` profile, which exercise the exact same code path minus the real network call. Run these
yourself, with a working `GEMINI_API_KEY` and normal internet access.

## What this measures, and why rules-only numbers are not enough

Everything in `HOLDOUT2_RESULT.md`, `HOLDOUT3_RESULT.md` and the main report is the **rule-based safety net with no
AI model** (`offline`/`checker_for('offline', ...)`). That is deliberate — it is the worst case, what a person gets
if the AI is unavailable or turned off — but it is *not* a measurement of how Quick Aid behaves in its normal,
AI-enabled mode, where the model does most of the understanding and the rules are a floor underneath it. Only a
`--model live` run tells you that. **Do not present the rules-only numbers as if they describe the AI-enabled
experience** — they describe the fallback, not the default.

## One-time setup

```bash
export GEMINI_API_KEY=...        # a real, working key
cd Quick-Aid
pip install -r requirements.txt
```

## Running it

```bash
# The full curated set (fast first check; ~100 cases, same set the offline report uses)
python evaluation/run_evaluation.py --model live --sets symptom_cases --label "live v1"

# The two generalisation sets, run against the real model. These are the numbers that matter most: do NOT edit
# any rule, prompt, or these datasets based on what you see here until you have looked at every failure (see below) -
# that is exactly how symptom_holdout.json and symptom_holdout2.json stopped being fresh estimates.
python evaluation/run_evaluation.py --model live --sets symptom_cases,symptom_holdout,symptom_holdout2,symptom_holdout3 \
    --label "live v1, all sets" --record evaluation/recordings/live_v1.jsonl
```

`--record FILE` saves every real request/response to a JSONL file. Do this on every live run: it lets you
**re-score the exact same responses later with zero API calls** (`--model replay --replay FILE`), which is how you
separate "the rules changed" from "the model's answers changed" when comparing two results.

Each run prints a summary and is saved for the dashboard (`evaluation/runs/`; `--no-write` skips saving). What you
get, matching every number requirement 2 asked for:

| What you asked for | Where it is |
|---|---|
| Emergency recall | `emergency_recall` |
| False negatives | `false_negative_rate` (= 1 − recall) |
| False reassurance | `false_reassurance_rate` — **new this round**: how often the model's own raw text claimed reassurance ("nothing to worry about", "this will clear up on its own"...) on a case that genuinely needed caution. This is captured *before* the safety net rewrites the text (see `triage_engine.soften()` / `_REASSURANCE_RE`), so it measures the **model's** behaviour, not what the person ends up seeing — the person never sees the unsoftened text, because the net corrects it every time (checked separately, see below) |
| Output validity | `structured_output_validity` (telemetry: how often the model's JSON actually matched the schema) |
| Latency, failures, cost per analysis | `latency_p50_ms` / `latency_p95_ms`, `api_failure_rate` / `fallback_rate`, `cost_per_analysis_usd` |
| Whether the safety engine corrects unsafe responses | `safety_intervention_rate`, plus the 30-case guardrail set (`model_output_cases.json`, run automatically as part of every report) — see below |

## Comparing prompt or model versions

```bash
# baseline
python evaluation/run_evaluation.py --model live --sets symptom_cases,symptom_holdout2,symptom_holdout3 --label "prompt v1" 
# ... change the prompt in symptom_checker.py / medical_analyzer.py, or switch MODEL_NAME to a different model ...
python evaluation/run_evaluation.py --model live --sets symptom_cases,symptom_holdout2,symptom_holdout3 --label "prompt v2: stricter chest-pain wording"

python evaluation/run_evaluation.py --list-runs
python evaluation/run_evaluation.py --compare <run_id_A> <run_id_B>
```

`--compare` flags any metric that got *worse* (lower recall, higher under-triage, higher false-reassurance, higher
latency/cost) so a prompt change that quietly regresses safety cannot slip through unnoticed. Every saved run
records the exact prompt fingerprint and dataset versions it used (`prompt_registry.fingerprint()`), so a
comparison is only ever between genuinely comparable runs.

## Does the safety engine actually correct unsafe AI responses?

Two complementary checks, both already passing against *scripted* bad responses (not yet against what the real
model actually says):

1. **The guardrail set** (`evaluation/datasets/model_output_cases.json`, 30 cases, `G01`–`G30`) feeds a deliberately
   wrong or dangerous model response (model says "low risk, nothing to worry about" for a clear emergency) through
   the *real* parsing and safety code, and checks the engine corrects it. This round added `G25`–`G30` specifically
   for the categories requirement 1 named: poisoning, severe allergic reaction, unconsciousness, breathing
   difficulty. Run it alone with:
   ```bash
   python -c "
   import sys; sys.path.insert(0, 'evaluation')
   import run_evaluation as ev
   r = ev.run_guardrails(ev.load('model_output_cases.json')['cases'])
   print(r['pass_rate'], r['failures'])
   "
   ```
2. **The AI-plus-triage workflow test** (`tests/test_emergency_detection_v2.py::TestAIPlusTriageWorkflow`) runs a
   *realistic, mostly-correct* mock model (not a deliberately broken one) through the real `/analyze_symptoms`
   pipeline, including one case where the model makes a plausible mistake (misses a stroke sign), and checks that:
   the model's correct answers pass through unharmed (the net does not second-guess a right answer), the one wrong
   answer gets corrected, and combined recall across the batch is complete. Run it with:
   ```bash
   pytest tests/test_emergency_detection_v2.py -k AIPlusTriageWorkflow -v
   ```

Neither of these touches the real model. What they do **not** tell you is whether the **real** Gemini model's
actual mistakes look like the ones scripted here, or something else entirely — only a live run can show that.
After a live run, look at every case in `failures` by hand (not just the aggregate numbers): if the real model's
failure mode is some new pattern of wrong answer, write new guardrail cases for it and re-run, the same way `G01`
through `G30` were each written for a specific observed or anticipated failure pattern.

## A realistic schedule

* **Before any production rollout:** `--model live --sets symptom_cases,symptom_holdout2,symptom_holdout3` once. Read every failure by hand.
* **On every prompt or model change:** the same, with `--record`, then `--compare` against the last saved run.
* **Periodically (e.g. monthly) in production:** re-run against the frozen held-out sets to catch silent drift if
  Google changes the underlying model without you changing anything. The sets are frozen specifically so this
  comparison stays meaningful over time.
* **When you have budget for it:** write a `symptom_holdout4.json` the same way `symptom_holdout3.json` was written
  (see its `limitations` field) — fresh wording, run once, no tuning in response to it — to keep getting an honest
  read on generalisation rather than slowly tuning against the same handful of sets.

## Cost note

Each case is one real API call (interview-mode cases are several). The curated set is ~100 cases; all four
symptom sets together are ~240. At current Gemini Flash pricing this is a small cost, but it is a real cost, so
`--no-write` plus `--sets symptom_cases` (the smallest, cheapest set) is the right choice for a quick sanity check
before spending on the full run.

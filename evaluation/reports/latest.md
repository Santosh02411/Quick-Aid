# Quick Aid evaluation report

Generated 2026-10-16 19:20 · engine v2

> **Read this first.** Written by the developers, NOT clinician-validated, and small (about 100 cases). It is a regression and engineering check on the safety rules, not evidence of clinical accuracy. Do not use these numbers to claim the app is safe or accurate for real patients.

## 1. Triage rules on curated symptom cases (no AI model)

97 scored cases (29 true emergencies).

| Metric | Result | Threshold |
|---|---|---|
| Emergency recall | 100.0% | >= 100.0% |
| Severe under-triage | 0.0% | <= 0.0% |
| Under-triage | 0.0% | <= 5.0% |
| False alarms | 0.0% | <= 10.0% |
| Over-triage | 0.0% | - |
| Abstention | 20.6% | - |
| Unsafe content | 0.0% | <= 0.0% |

**Pass rate by category**

| Category | Cases | Pass |
|---|---|---|
| adversarial | 5 | 100.0% |
| emergency | 27 | 100.0% |
| interview | 12 | 100.0% |
| negation_trap | 8 | 100.0% |
| routine | 14 | 100.0% |
| see_soon | 10 | 100.0% |
| urgent_same_day | 16 | 100.0% |
| vague | 5 | 100.0% |

**Does reliability track correctness?** (confidence labels are never used as probabilities)

| Reliability shown to user | Cases | Clean | Under-triage |
|---|---|---|---|
| good | 2 | 100.0% | 0 |
| limited | 83 | 100.0% | 0 |
| none | 12 | 100.0% | 0 |

No failures on the scored cases.

**Known limitations** (reported, not scored):

- `T09` 'the chest pain went away last week, today I just have a sore throat': level=high, alert=True, FALSE_ALARM, OVER_TRIAGE

## 2. Guardrails against unsafe model output

24/24 cases passed (100.0%). Each case feeds a deliberately unsafe, overconfident or malformed model answer through the real parsing and triage code and checks what would be displayed.


## 2b. Generalisation: held-out sets (informational, not gated)

The main set above was written while the rules were being tuned, so it is optimistic. These sets measure how the rules cope with new wording.

| Set | Cases | Emergency recall | Severe under-triage | False alarms | Abstention |
|---|---|---|---|---|---|
| Quick Aid HELD-OUT symptom cases | 43 | 100.0% | 0.0% | 0.0% | 39.5% |
| Quick Aid HELD-OUT set 2 (fresh, written after the first hol | 50 | 30.8% | 50.0% | 0.0% | 56.0% |

`symptom_holdout.json` is no longer unseen (its misses were fixed afterwards; first-run result: `HOLDOUT_FIRST_RUN.md`). `symptom_holdout2.json` is the fresh estimate (`HOLDOUT2_RESULT.md`). **Without an AI model the rules miss most emergencies phrased in unfamiliar words or other languages.**

## 3. What the triage engine adds to an UNSAFE model

A simulated AI that answers "low urgency, common cold" to everything. The engine can only help where a rule fires, so the fresh set is the fairer row:

| Cases | Emergency recall: model alone | with engine | Under-triage: model alone | with engine |
|---|---|---|---|---|
| main set (in-sample) | 0.0% | 100.0% | 100.0% | 17.0% |
| fresh held-out set 2 | 0.0% | 30.8% | 100.0% | 68.4% |

In other words: if the AI is wrong and the wording is unfamiliar, the rules often cannot rescue it. The engine is a floor, not a second opinion.

## 4. Photo-quality checks (synthetic images)

Rating accuracy 100.0%; unusable photos caught 100.0%; good photos wrongly rejected: 0.


## Thresholds

**All safety thresholds met.**

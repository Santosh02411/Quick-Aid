# Held-out set 4: first-run result (rules not changed in response to this run — by explicit instruction)

`symptom_holdout4.json` specifically targets three categories named in a later round of this project's work — children, pregnancy, and
older adults — with several cases grounded in independently published clinical criteria rather than the developer's own judgement, plus
contradictory/hedging symptom descriptions and realistic misspellings. It was run **once**. Unlike `symptom_holdout2.json` and
`symptom_holdout3.json` before it (each used to fix the specific gaps it found), **no rule was changed in response to this run's specific
failures**, per an explicit instruction given for this round: keep a genuinely untouched test set.

## External sources used for ground truth

- **Children**: WHO/UNICEF IMCI (Integrated Management of Childhood Illness) general danger signs for ages 2 months–5 years: inability to
  drink or breastfeed, vomiting everything, convulsions, lethargy or unconsciousness.
- **Pregnancy**: WHO Antenatal Care Guideline Implementation Guide danger signs, and the Government of India's published list (via
  vikaspedia.in) of antenatal/intranatal/postnatal danger signs: vaginal bleeding, convulsions, severe headache/blurred vision, water
  breaking without labour pain, labour lasting more than 12 hours, retained placenta, loss of fetal movement, high fever, foul-smelling
  discharge.
- **Older adults**: NHS guidance on falls (urgent assessment for head/neck/back/hip injury) and NHS sepsis red flags (collapsed/unrousable,
  no pulse, breathing very fast, blue lips, new non-fading rash or mottled skin, no urine in 12 hours), plus published geriatric-medicine
  reporting that infection in older adults often presents **without the classic fever** — as sudden confusion or falls instead.

## Result (rule-based triage only, no AI model)

- Cases: 38 (27 true emergencies)
- **Emergency recall: 22.2%** (6 of 27)
- Severe under-triage: 46.7%; under-triage: 70.0%
- False alarms: 0.0%; abstention: 52.6%

## What this means

This is a substantially worse result than either held-out set before it (which, after their own fixes, both now score 100%). That is the
expected, honest shape of the finding, not a surprise to be explained away: **the previous two rounds of fixes closed gaps in the wording
they were shown — colloquial/idiomatic general emergency phrasing — and that improvement does not transfer to categories with their own
distinct vocabulary that was never specifically targeted before now.** Pediatric, obstetric and geriatric danger signs use clinical
language ("won't breastfeed", "waters broke", "retained placenta", "hasn't passed urine") that is simply absent from the current pattern
library, which was built around general adult emergency presentations (chest pain, breathing difficulty, stroke, bleeding, poisoning).

**What was caught:** general presentations that happen to overlap with already-covered categories still worked even in these contexts —
confusion with falls (H420, via the existing fever-with-confusion—adjacent wording), blue lips and fast breathing (H421, via the existing
breathing-difficulty pattern), a child with fever and confusion (H408, via the existing confusion pattern), and the contradictory/hedging
cases that still contained a directly-recognised symptom word regardless of the surrounding uncertainty (H426, H430, H436) — showing the
existing hedging-language robustness (built earlier, for a different purpose) generalises to these new phrasings. False alarms stayed at
0%, consistent with every set so far.

**What was missed, overwhelmingly:** the vocabulary specific to these three categories. None of "won't breastfeed", "vomiting everything",
"floppy", "waters broke" + preterm, "felt the baby move", "labour pains" + duration, "placenta hasn't come out", "passed urine", or a
*typo-laden* version of an otherwise-recognisable phrase ("tihgt", "braethe", "collapsd", "weking", "swallowd a hole bottel") are
recognised at all right now.

## Why this is being left unfixed for now

Fixing these now, in direct response to seeing this specific run's failures, would repeat exactly the pattern that made `symptom_holdout2`
and `symptom_holdout3` no longer trustworthy as fresh estimates after their own fix rounds — and would mean this project has never once
had an honest, multi-category measurement that was allowed to simply stand. The instruction for this round was explicit on this point, and
it is being honoured: **this result is recorded as-is, and is the number to cite for "how does the rule-based floor do on children,
pregnancy and older adults" until a dedicated, separately-planned and separately-measured improvement round happens — ideally with the
same discipline this round asked for (fix, then measure against a *new*, not-yet-tuned set, not this one again).**

## What a future round should do

- Build a pediatric/obstetric/geriatric-specific pattern vocabulary (the specific missed phrases above are a direct starting list),
  informed by the WHO/NHS sources cited.
- Write a **held-out set 5**, fresh, before doing that work, so its result is not pre-contaminated by having been looked at during the fix.
- Specifically re-test the atypical-presentation case (confusion/falls without fever in an older adult) after any fix, since a naive fix
  that only adds "fever" to existing patterns would miss the entire clinical point of that example.
- Typo tolerance currently only covers a short, hand-reviewed list of general emergency terms (`TYPO_FIXES` / `_FUZZY_SAFE_TERMS` in
  `triage_engine.py`); it does not yet cover pediatric/obstetric/geriatric-specific vocabulary either.

## Misses

- `H401` "my 8-month-old won't breastfeed at all and just lies there not really reacting to anything": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H402` "the baby has been vomiting up everything we give her for the last few hours, can't keep an": MISSED_EMERGENCY, UNDER_TRIAGE
- `H403` "my 3 year old had a fit that lasted a couple of minutes, shaking all over, during this fev": MISSED_EMERGENCY, UNDER_TRIAGE
- `H404` "he's only 2 and so sleepy I can barely wake him up, he was fine yesterday": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H407` "newborn hasn't fed properly since this morning and feels floppy when I pick her up": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H409` "I'm 30 weeks pregnant and I just had a fit, I'm shaking all over": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H410` "my waters broke but I'm not due for another 6 weeks and I'm not having any contractions": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H411` "I'm 34 weeks and I haven't felt the baby move since yesterday": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H412` "severe headache and my vision keeps going blurry, I'm 8 months pregnant": MISSED_EMERGENCY
- `H413` "it's been 14 hours of labour pains and the baby still hasn't come": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H414` "the placenta hasn't come out and it's been over an hour since the baby was born": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H415` "I'm pregnant and there's a really foul smell coming from the discharge, plus I have a feve": MISSED_EMERGENCY, UNDER_TRIAGE
- `H418` "a little swelling in my ankles by the end of the day, 24 weeks pregnant": UNDER_TRIAGE
- `H419` "my 80 year old father fell and hit his head, he seems a bit out of it now": MISSED_EMERGENCY, UNDER_TRIAGE
- `H422` "grandfather hasn't passed urine in about 14 hours and seems very unwell and drowsy": MISSED_EMERGENCY
- `H423` "my 90 year old grandmother has a new purple rash that's spreading and she feels cold to th": MISSED_EMERGENCY, UNDER_TRIAGE
- `H427` "it's probably nothing but my left arm has gone numb and I also have a headache, not sure i": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H428` "maybe it's just heartburn, maybe it's my heart, there's a tight feeling in my chest that s": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H429` "the pain comes and goes, sometimes it's really bad sometimes it's nothing, it's in my ches": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H433` "chest is really tihgt and i cant braethe properly": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H434` "my hubby collapsd and is not weking up at all": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H435` "baby swallowd a hole bottel of medcine": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H438` "stomach is paining too much since yesterday, not able to eat anything": UNDER_TRIAGE

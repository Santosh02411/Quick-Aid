# Held-out set 3: first-run result (rules not changed in response to this run)

`symptom_holdout3.json` was written after `symptom_holdout2.json` had already been used to fix the rules (see `HOLDOUT2_RESULT.md`,
which took recall from 30.8% to 100% on that set). None of this set's wording reuses set 2's phrasing. It specifically targets the
categories asked for when this round of work began — poisoning, severe allergic reactions, unconsciousness, breathing difficulties —
plus real-script Hindi/Kannada/Marathi text (not just Latin transliteration), deliberate misspellings, and indirect/colloquial
descriptions meant to need the new combination-signal and typo-tolerance layers rather than a single matching phrase. It was run
once. **No rule was changed in response to this specific run before this result was recorded.**

## Result (rule-based triage only, no AI model)

- Cases: 48 (29 true emergencies, including 2 prompt-injection attempts)
- **Emergency recall: 37.9%** (11 of 29)
- Severe under-triage: 44.7%; under-triage: 55.3%
- False alarms: 0.0%; abstention: 52.1%

## What this means

- The rules generalised **partially**, not fully, to wording they were never tuned against. The fixes made in response to set 2 were
  real (they fixed exactly what they targeted, and set 2 still scores 100%), but **a new set of fresh phrasing finds new gaps** —
  exactly the pattern already seen once between set 1 and set 2. This is the expected, honest behaviour of a hand-written pattern
  library: each round of fixes closes the gaps it was shown, not the ones it wasn't.
- **What generalised well:** all 3 real-script non-English cases (Hindi/Devanagari, Kannada script, Marathi/Devanagari) were caught,
  the two prompt-injection cases were both caught, one of the two Latin-transliteration cases was caught, 2 of 3 deliberately
  misspelled cases were caught (the typo-tolerance layer worked for "colapsed"/"unconsious"/"breething", which are close to its
  curated list; it did not yet cover "weedkiler" or "siezed", which weren't).
- **What did not generalise:** indirect phrasing using synonyms the combination-signal rules did not anticipate ("went ghost white"
  instead of "pale", "crumpled to the ground" instead of "collapsed", "chest is heaving" instead of "trouble breathing"), accidental
  pediatric poisoning described by circumstance rather than an ingestion verb, and a few specific symptom phrasings (eye pain
  described as "sharp" rather than "severe", blood noticed "when I wipe" rather than "in my stool").
- As before: **false alarms stayed at 0%.** The rules are conservative in both directions - when unsure, they ask for more detail
  rather than guessing, but they also never raise an alarm on a negated or genuinely mild description.
- This confirms, with a second independent measurement, the project's standing conclusion: **a keyword-and-pattern scanner cannot be
  made complete by writing more patterns.** It is a last-resort floor under an AI model, not a substitute for one. Whether the live
  AI model does better on this exact set is not yet measured - see the note in `HOLDOUT2_RESULT.md` about running `--model live`.

## Roadmap (not done)

- A fourth fresh set, after the misses below are fixed, to see whether the SAME gap (new synonyms always being missed) keeps
  recurring - which would argue for a different approach (e.g. an embedding/similarity-based matcher) rather than more hand-written
  patterns.
- Have a clinician review and extend the datasets.
- Measure the live model on all four sets.

## Misses

- `H302` "she can't catch her breath at all and her chest is heaving like crazy": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H303` "the baby's breathing sounds really raspy and he's gone quiet and floppy": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H304` "I had shellfish and now my throat feels like it's closing up and I'm covered in hives": MISSED_EMERGENCY, UNDER_TRIAGE
- `H306` "my neighbour is slumped against the wall and nothing I do wakes her up": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H307` "he just dropped to the ground mid-sentence and hasn't moved since": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H308` "my toddler got into the cupboard, there's spilled dishwasher tablets everywhere and some packets are open": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H309` "he necked a load of his grandmother's heart tablets, says he didn't want to wake up": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H310` "she accidentally sprayed bug spray right into her mouth while playing with the can": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H312` "my mother went ghost white, grabbed the table and crumpled to the ground": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H313` "half his face has dropped and his words are coming out all jumbled": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H315` "35 weeks along and I'm bleeding through more than one pad an hour": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H316` "newborn's gone floppy and won't latch, her skin looks mottled and blotchy": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H317` "sliced his hand open on a broken bottle and blood's soaking through every towel we have": MISSED_EMERGENCY (reached 'high' via another route, but the emergency alert itself did not fire)
- `H318` "the ladder gave way and she's not been able to move since she landed": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H319` "swallowed a whole strip of her antidepressants, left a note on the table": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H323` "bachhe ne dawai ki puri bottle pi li hai" (Hindi, Latin script: child drank the whole bottle of medicine): MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H326` "she drank the weedkiler by accident and is vomitting" (misspelt "weedkiller"): MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H327` "he siezed for over a minute and his lips went purple" (misspelt "seized"): MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H332` "a new lump showed up in my armpit about a week ago": SEVERE_UNDER_TRIAGE
- `H333` "ringing in my ears that won't go away, been like this for 5 days": UNDER_TRIAGE
- `H334` "sharp pain behind my left eye that comes and goes": UNDER_TRIAGE
- `H335` "blood when I wipe after going to the toilet, on and off for a week": UNDER_TRIAGE

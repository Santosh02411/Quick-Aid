# Held-out set 2: result after the rules were fixed (run once, rules NOT changed afterwards)

`symptom_holdout2.json` was written after the miss families from the first held-out set were fixed, using fresh and more realistic phrasing
(slang, indirect descriptions, typos, and a few transliterated Hindi/Kannada phrases). It was run once. **No rule was changed in response**, so this is the best available estimate of how the rule-based safety net generalises.

## Result (rule-based triage only, no AI model)

- Cases: 50 (26 true emergencies)
- **Emergency recall: 30.8%** (8 of 26); non-English emergencies caught: 0 of 4
- Severe under-triage: 50.0%; under-triage: 63.2%
- False alarms: 0.0%; over-triage: 0.0%; abstention: 56.0%

## What this means

- The rules are **precise but narrow**: no false alarms and no over-triage, but they only recognise wordings they were written for. Fixing the first held-out set's misses did not generalise to new wordings.
- A keyword scanner cannot be made complete. It is a **last-resort floor under the AI model**, not a way to detect emergencies on its own.
- **Without an AI model (basic mode) the app will miss most emergencies that are not described in common English wording.** The interface now says this plainly, keeps the emergency number visible during every check, and never answers an unrecognised description with reassurance.
- Whether the AI model does better is **not yet measured**. Run `python evaluation/run_evaluation.py --live` with a real `GEMINI_API_KEY` to measure it on these same cases.

## Roadmap (not done)

- Measure the live model on all sets and publish the numbers.
- Have a clinician review and extend the datasets; add more languages.
- Consider a second AI pass dedicated only to emergency detection.

## Misses

- `H202` "he just keeled over at the gym and won't respond": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H204` 'ate peanuts and now my face is swelling and I feel like my airway is tight': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H205` "severe crushing pressure in the middle of my chest, I'm sweating and nauseous": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H206` 'she drank some weedkiller by mistake': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H208` 'my uncle suddenly went pale, clutched his left arm and fell down': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H209` 'toddler found with an empty bottle of tablets next to her': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H210` 'I took a lot of painkillers because I just want everything to stop': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H211` "I keep seeing things that aren't there and I can't stop shaking and my fever is huge": MISSED_EMERGENCY, UNDER_TRIAGE
- `H212` "my son's having an asthma attack and can barely talk": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H214` 'she has been having fits for ten minutes': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H216` "I'm 8 months pregnant and there's a gush of blood": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H217` "my chest is on fire and my left shoulder aches, I'm drenched in sweat": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H219` 'the baby is making grunting noises and her ribs are sucking in with each breath': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H220` "he's turned a funny colour and isn't responding when I shout": MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H221` 'seene me bahut tez dard hai aur paseena aa raha hai': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H222` 'saans nahi le pa raha': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H223` 'ede novu thumba ide': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H224` 'mera bacha behosh ho gaya hai': MISSED_EMERGENCY, SEVERE_UNDER_TRIAGE
- `H226` "I'm 37 weeks and my waters broke and the fluid is green": SEVERE_UNDER_TRIAGE
- `H227` 'I got a deep cut on my hand and I can see the tendon': SEVERE_UNDER_TRIAGE
- `H228` "my 4 year old has a red hot swollen knee and won't walk and has a fever": UNDER_TRIAGE
- `H231` 'burning wee for three days': UNDER_TRIAGE
- `H234` 'coughing for 4 weeks now': UNDER_TRIAGE
- `H235` 'I keep fainting when I stand up': UNDER_TRIAGE

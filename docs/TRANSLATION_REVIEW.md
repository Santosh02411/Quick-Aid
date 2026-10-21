# Reviewing the Hindi, Marathi and Kannada text

**Status: not reviewed by a native-speaking clinician.** All non-English text was written by an AI assistant. An AI assistant has since also done a careful
*reading-level linguistic pass* over it (see "What the AI pass covered" below) and fixed real problems it found — but **that pass is not a substitute for the
human review this page asks for**, and does not change the on-screen status: no language here is marked `reviewed_by_clinician`, and none should be until an
actual qualified person signs off. An AI reviewing its own (or another AI's) output cannot catch what only lived clinical and cultural experience would —
whether "सीपीआर" means anything to someone who has never heard the English term, whether a phrase that reads fine on screen sounds patronising or alarming
spoken aloud, or whether the chosen first-aid steps match what a doctor in India would actually want a bystander to do before the ambulance arrives.

## What the AI pass covered, and what it found

A later development round had an AI assistant (not the one that originally wrote the translations) reread every Hindi/Marathi/Kannada string in `catalog.py`
and `emergency.py` for medical accuracy, natural phrasing and internal consistency, and separately stress-test the **detection** side with colloquial, idiomatic
and negated phrasing a reviewer would not necessarily think to type. This found and fixed real issues, recorded here so a human reviewer does not have to
re-discover them, and so the gap between "mechanically tested" and "medically reviewed" stays honest:

* A **false-alarm bug**: common intensifier words between a symptom and its negation ("बिल्कुल नहीं" = not at all, "...तो नहीं" = an emphatic no, Marathi
  "अजिबात नाही", Kannada "ಖಂಡಿತ ಇಲ್ಲ") broke negation detection, so an ordinary, unambiguous denial read as a positive match. Fixed with a short, curated
  filler-word list (deliberately not "any word", to keep detection erring toward raising an alarm rather than suppressing one).
* **7 colloquial/idiomatic gaps**, found by testing phrasing nobody had tried yet rather than the dataset's own wording: "the chest is bursting" (छाती फट रही
  है / छाती फुटल्यासारखी / ಎದೆ ಒಡೆದ ಹಾಗೆ), "breath coming apart" (सांस उखड़ रही है), "darkness before the eyes" for fainting (Hindi, Marathi and Kannada
  versions), and a scorpion "striking with its sting" (नांगी मारली) rather than "biting".
* An **understatement gap**: "थोड़ा सा भारीपन" (a little heaviness) did not match, because the pattern only allowed an *amplifying* qualifier ("very sharp"),
  not a minimising one — yet people often understate a real cardiac symptom exactly this way. Fixed.
* One **minor Kannada transliteration** nit in a non-safety-critical accessibility label ("ಸ್ಕ್ರಾಲ್" → "ಸ್ಕ್ರೋಲ್" for "scroll").
* Everything else read as medically accurate and natural on this pass: the first-aid steps (recovery position, nothing by mouth during a seizure, no induced
  vomiting, no tourniquets, time-of-onset noted for stroke), the consent notice, and the emergency-page summaries. "Read as accurate to an AI with Hindi/Marathi/
  Kannada knowledge but no medical license or lived native-speaker judgement" is exactly as much weight as that should carry — which is why it still is not
  a pass.

The AI pass was strongest in Hindi, good but slightly less certain in Marathi (shared Devanagari script, closely related grammar), and most limited in Kannada,
where its confidence is lower and a human reviewer should look hardest first.

## What tests already guarantee (so reviewers can focus on meaning)

* Every string exists in all four languages; placeholders (`{number}`, `{n}`) are identical; emergency numbers are placeholders, never typed digits that could be altered.
* Every translated warning sign has the **same number of steps** as English, and the first step always tells the person to call.
* The important "do **not**" instructions are present (nothing in the mouth during a seizure, do not induce vomiting, do not cut or suck a snake bite, do not drive yourself…).
* Text is in the right script, with no stray English words; phone and helpline numbers use ordinary digits (so they can be dialled as printed).
* Warning-sign detection fires for about 85 example phrases (native script and Roman letters, including idiomatic and understated phrasing), stays silent for
  20+ everyday complaints and negated statements (including intensified negation: "बिल्कुल नहीं", "अजिबात नाही", "ಖಂಡಿತ ಇಲ್ಲ"), and resolves "no chest pain" correctly.
* A symptom's own stated resolution ("the chest pain went away last week") is not an alarm in English; the hi/mr/kn detector does not yet have an equivalent
  mechanism (see `TestResolvedSymptomsAcrossLanguages` in `tests/test_i18n.py`, which documents this gap rather than hiding it) - worth a reviewer's attention.

## What only a human can judge

Review **each item below**, by a native speaker **and** a clinician (ideally two different people), and write the result in `quickaid/i18n/translation_status.json`.

1. **Medical accuracy of the instructions** in `quickaid/i18n/emergency.py` `GUIDANCE`: first-aid steps for chest pain, breathing difficulty, stroke, seizure, bleeding, anaphylaxis, severe headache,
   poisoning, meningitis signs, confusion with fever, snake/scorpion bite, serious injury; the default steps; the self-harm steps and the helpline text (confirm the Tele-MANAS number 14416 is current).
2. **Terminology people actually use**, not textbook terms: is "स्ट्रोक (लकवा)" understood? Is "सीपीआर" understood, or should it be explained? Which Kannada word for "seizure" is least ambiguous?
3. **Register and tone**: calm, direct, not frightening, respectful, understandable at a low reading level.
4. **Detection patterns** in `emergency.py` `_RAW`: add phrases and spellings people really use (including regional and colloquial ones, and Roman-letter spellings), and say which are too broad.
   Every pattern you add should come with a test example in `tests/test_i18n.py`.
5. **The consent text** (`consent.*`) and the AI-data explanation: would an ordinary person understand exactly what is sent and to whom?
6. **The urgency headlines** (`js.*_title`, `js.*_msg`): "see a doctor today" must not read as "optional".
7. **The 15 first-aid guides** (`quickaid/ai/first_aid.py`) and **the first-aid guides on the emergency page** are currently English-only (the page says so). Translating them is future work and needs the same review.

## Recording the review

Edit `quickaid/i18n/translation_status.json`:

```json
{ "reviewed_by_clinician": {"hi": true, "mr": false, "kn": false},
  "reviewers": [{"language": "hi", "name": "Dr A. Example", "role": "GP", "date": "2026-11-02", "scope": "emergency.py GUIDANCE + catalog em.*"}],
  "last_review": "2026-11-02" }
```

A language's on-screen "not yet reviewed" notice should only be removed after **both** a clinician and a native speaker have approved *all* safety-critical files,
and after the tests (`pytest tests/test_i18n.py`) pass with the reviewers' changes.

## Changing a translation later

Any edit to a safety-critical file resets that language's review status to `false`. The tests will fail if you add a string without all four languages.

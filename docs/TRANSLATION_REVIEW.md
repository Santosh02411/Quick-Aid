# Reviewing the Hindi, Marathi and Kannada text

**Status: not reviewed.** All non-English text was written by an AI assistant. It has been tested for *mechanical* problems (see below) but **no native-speaking
clinician has signed it off**. Emergency instructions can hurt people if they are wrong, so the app shows a notice on every translated page and result, and this
checklist exists so the review can be done properly and recorded.

## What tests already guarantee (so reviewers can focus on meaning)

* Every string exists in all four languages; placeholders (`{number}`, `{n}`) are identical; emergency numbers are placeholders, never typed digits that could be altered.
* Every translated warning sign has the **same number of steps** as English, and the first step always tells the person to call.
* The important "do **not**" instructions are present (nothing in the mouth during a seizure, do not induce vomiting, do not cut or suck a snake bite, do not drive yourself…).
* Text is in the right script, with no stray English words; phone and helpline numbers use ordinary digits (so they can be dialled as printed).
* Warning-sign detection fires for about 70 example phrases (native script and Roman letters), stays silent for 17 everyday complaints, and understands "no chest pain".

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

# Reference images for evaluating REAL model output

Quick Aid ships **no real medical photos** (privacy, consent and licensing). To measure how the AI really behaves on
photographs, you supply a small reference set. This is what makes the measurement *independent*:

* **Labels must not come from the model or from the person who writes the prompts.** Ideally a clinician labels them;
  say so honestly in `clinician_reviewed`.
* **Keep a `test` split that nobody tunes against.** Iterate prompts on `dev`; look at `test` rarely, and never change a
  prompt *because of* a test result without declaring the test split used up (and starting a new one).
* **Record provenance** for every image: where it came from, its licence, who labelled it. The loader refuses images without it.
* Thirty images is the *minimum* for any statement at all, and still small. The report says so.

## Where the photos can come from

* Public, licensed collections (check each licence; many dermatology sets forbid redistribution, so keep them out of git).
* Your own photographs, with the consent of everyone in them, never identifiable faces or tattoos.
* **Never** use real user uploads. Quick Aid deletes them immediately, and it must stay that way.

## Format: `evaluation/reference/manifest.json`

```json
{
  "images": [
    {
      "file": "images/graze_01.jpg",
      "split": "dev",
      "source": "own photo, consent given",
      "license": "CC0",
      "labelled_by": "Dr A. Example, GP",
      "clinician_reviewed": true,
      "expected_image_type": "injury_or_wound",
      "expected_min_urgency": "low",
      "expected_max_urgency": "low",
      "expected_finding_keywords": ["graze", "red"],
      "must_be_unusable": false
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `expected_min_urgency` / `expected_max_urgency` | the least / most urgent answer that would be acceptable (under-triage is the dangerous error) |
| `expected_image_type` | what the photo is (`injury_or_wound`, `burn`, `bruise_or_swelling`, `skin_condition`, `xray_or_scan`, `not_medical`, ...) |
| `expected_finding_keywords` | words a good description of what is **visible** should contain (scored as recall) |
| `must_be_unusable` | the photo is too blurry/dark/etc.: the system must ask for a better one, not guess |

## Running it

```bash
# against the real model, recording every response so the run can be re-scored later without spending money
python evaluation/run_evaluation.py --model live --image-reference evaluation/reference --record evaluation/recordings/run1.jsonl --label "image prompt v3"

# re-score the same recorded responses after changing the safety rules or metrics (no API calls)
python evaluation/run_evaluation.py --model replay --replay evaluation/recordings/run1.jsonl --image-reference evaluation/reference
```

Reported: under-triage, severe under-triage, over-triage, unusable-photo recall, image-type accuracy, finding-keyword recall,
and **interpretations offered with no visible findings** (which should be zero). The dashboard shows them next to the
run, with caveats about set size and clinician review.

`images/` is git-ignored on purpose. Only the manifest is meant to be shared.

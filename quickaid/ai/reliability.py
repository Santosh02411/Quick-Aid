"""Reliability and uncertainty communication.

IMPORTANT: this is NOT a probability that the result is correct. A model's own
"confidence" label is not calibrated against real outcomes, so we never show it as a
percentage or as the chance of being right. Instead we describe, in plain words, how
much information the result was based on and how well it could be checked:

    limited      - little information, no AI, a poor photo, or the safety rules disagreed with the AI
    moderate     - reasonable information with some gaps
    well-informed - good information, but still not a diagnosis

Every result carries a statement that it is an estimate and not a diagnosis, no matter
how good the reliability looks.
"""

from typing import Dict, List, Optional

LABELS = {
    'limited': 'Limited information',
    'moderate': 'Moderate information',
    'good': 'Reasonably well-informed',
}

NOTE = ('This describes how much information the result was based on and how well it could be checked. '
        'It is not the chance that the result is correct.')

AI_RATING_NOTE = ("The AI's own confidence rating is a rough self-assessment. It has not been verified against "
                  "real outcomes, so please do not read it as a probability.")

STATEMENTS = {
    'limited': ('This is a rough estimate based on limited information. It is not a diagnosis, and it may well be wrong. '
                'If you are worried, please see a healthcare professional.'),
    'moderate': ('This is an estimate based on what you shared. It is not a diagnosis and it cannot replace an examination. '
                 'A healthcare professional can tell you what is really going on.'),
    'good': ('This estimate used the information you gave, but it is still not a diagnosis and cannot replace an examination. '
             'Please see a healthcare professional if you are unsure.'),
}


def assess(kind: str, *, mode: str, ai_self_rating: Optional[str] = None, image_quality: Optional[str] = None,
           answered: Optional[int] = None, skipped: Optional[int] = None, interview: bool = False,
           input_chars: Optional[int] = None, symptoms_found: Optional[int] = None,
           safety_overrode: bool = False, insufficient: bool = False, explanations: int = 1) -> Dict:
    """Describe how reliable a result is. Never returns a number."""
    points = 4
    reasons: List[str] = []

    def lose(n: int, why: str):
        nonlocal points
        points -= n
        reasons.append(why)

    if insufficient:
        lose(4, 'There was not enough usable information to assess this properly.')
    if mode in ('basic', 'rules'):
        lose(3, 'No AI model was used. This is a rule-based estimate.')

    if kind == 'image':
        if image_quality == 'poor':
            lose(3, 'The photo quality is poor, so details may be missed or misread.')
        elif image_quality == 'fair':
            lose(1, 'The photo is not ideal (lighting, focus or size), which limits what can be seen.')
        if ai_self_rating == 'low':
            lose(2, 'The AI itself rated its confidence as low.')
    else:
        if interview:
            total = (answered or 0) + (skipped or 0)
            if answered is not None and answered == 0:
                lose(3, 'None of the follow-up questions were answered.')
            elif answered is not None and skipped is not None and skipped > answered:
                lose(2, 'Most of the follow-up questions were skipped, so many details are unknown.')
            elif skipped and total and skipped * 3 >= total:
                lose(1, 'Some follow-up questions were skipped, so some details are unknown.')
        else:
            lose(1, 'Based on a short description without follow-up questions.')
        if input_chars is not None and input_chars < 25:
            lose(1, 'The description was very short.')
        if symptoms_found == 0:
            lose(1, 'No specific symptoms could be identified.')
        if ai_self_rating == 'low':
            lose(2, 'The AI itself rated its confidence as low.')

    if safety_overrode:
        lose(1, "Our safety rules raised the urgency above the AI's first estimate, so the two disagreed.")
    if explanations == 0 and not insufficient:
        lose(1, 'No specific explanation could be suggested.')

    level = 'good' if points >= 3 else 'moderate' if points == 2 else 'limited'
    if not reasons and level == 'good':
        reasons.append('You gave enough detail for a reasonable estimate, but it is still only an estimate.')
    out = {'level': level, 'label': LABELS[level], 'reasons': reasons, 'note': NOTE}
    if ai_self_rating:
        out['ai_self_rating'] = {'value': ai_self_rating, 'note': AI_RATING_NOTE}
    return out


def uncertainty_statement(level: str) -> str:
    return STATEMENTS.get(level, STATEMENTS['limited'])

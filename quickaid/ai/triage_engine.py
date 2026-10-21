"""Centralized medical triage and safety engine.

Every safety decision in Quick Aid lives here, so there is ONE place to read, test and
improve it. The symptom checker, the adaptive interview, the image analyzer and the
follow-up chat all hand their results to this module before anything is shown or saved.

What it does
------------
1. EMERGENCY DETECTION   - scans free text (negation-aware) for emergency, urgent and
                           self-harm language, and returns immediate step-by-step guidance.
2. URGENCY CLASSIFICATION - turns evidence into a level (low / medium / high) and a care
                           level (self-care / soon / same-day / emergency) with a timeframe.
3. SAFETY VALIDATION      - reviews AI output before display:
                             * urgency can only be RAISED by safety rules, never lowered
                             * serious conditions or visible danger signs in the model's own
                               text force a minimum urgency
                             * overconfident wording ("you definitely have...") is softened
                             * medicine doses and "stop your medication" advice are removed
                             * reassurance that contradicts the urgency is corrected
4. HONEST COMMUNICATION   - every reviewed result carries reliability, an uncertainty
                           statement, what could not be assessed, and a log of every change
                           the safety rules made (so nothing happens silently).

The engine never diagnoses. Rules are conservative on purpose: when in doubt it escalates.
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from quickaid.ai import reliability
from quickaid import i18n
from quickaid.i18n import emergency as i18n_emergency
from quickaid.core.localization import localize_text

ENGINE_VERSION = 2

LEVELS = ('low', 'medium', 'high')
RANK = {'low': 1, 'medium': 2, 'high': 3}
NAME = {1: 'low', 2: 'medium', 3: 'high'}
N = '{EMERGENCY_NUMBER}'

CARE_FOR_LEVEL = {'low': 'self_care', 'medium': 'soon', 'high': 'same_day'}
DURATION_RANK = {'hours': 0, 'days_1_3': 1, 'days_4_7': 2, 'weeks_1_2': 3, 'weeks_2plus': 4}
DURATION_LABEL = {'hours': 'Less than a day', 'days_1_3': '1–3 days', 'days_4_7': '4–7 days',
                  'weeks_1_2': '1–2 weeks', 'weeks_2plus': 'More than 2 weeks'}
AGE_LABEL = {'infant': 'A baby (under 1 year)', 'child': 'A child (1–12 years)', 'teen': 'A teenager (13–17)',
             'adult': 'An adult (18–64)', 'older': 'An older adult (65 or over)'}

DISCLAIMER_AI = ('AI-generated estimate for general information. It is not a diagnosis, it cannot examine you, '
                 'and it can be wrong. Always consult a healthcare professional.')
INTERPRETATION_NOTES = {
    'unvalidated_ai': ('These explanations are an AI model\'s guesses based on what it could see or was told. They have not been validated '
                       'against clinical outcomes and may be wrong or incomplete.'),
    'rules': 'These come from simple built-in rules, not medical expertise. They have not been clinically validated.',
}
DISCLAIMER_RULES = ('Rule-based estimate, not a diagnosis. It cannot examine you and can be wrong. '
                    'Always consult a healthcare professional.')

# ---------------------------------------------------------------------------
# 1. Emergency detection
# ---------------------------------------------------------------------------

_NEGATORS = re.compile(r"\b(no|not|without|never|denies|deny|isn'?t|aren'?t|don'?t have|doesn'?t have|free of|ruled out)\b")
_CLAUSE_BREAKS = ('.', ';', '!', '?', ',', ' but ', ' just ', ' only ', ' except ', ' however ', ' although ')


def negated(text: str, start: int) -> bool:
    """True if a negation word appears earlier in the same clause as the match.

    Clauses end at punctuation or words like "but"/"just"/"only", so "no fever, just a cough"
    negates the fever but not the cough.
    """
    clause_start = -1
    for b in _CLAUSE_BREAKS:
        i = text.rfind(b, 0, start)
        if i != -1:
            clause_start = max(clause_start, i + len(b) - 1)
    window = text[max(clause_start + 1, start - 40):start]
    return bool(_NEGATORS.search(window))


# (label, regex). A non-negated match means "call emergency services".
EMERGENCY_PATTERNS: List[Tuple[str, str]] = [
    ('Chest pain or pressure',
     r"chest\s+(pain|pressure|tightness|discomfort|hurts?)|pain in (my |the )?chest|tight chest|heart attack|"
     r"chest (feels|is|feel) (tight|heavy|squeezed|crushed|like)|squeez\w+ (in|around|on) (my |the )?chest|"
     r"(heaviness|tightness|pressure) (in|on) (my |the )?chest|elephant on (my )?chest|crushing (pain|weight)"),
    ('Severe trouble breathing',
     r"short(ness)? of breath|can'?t breathe|cannot breathe|(trouble|difficulty|hard to|struggling to|unable to) breathe?(ing)?|"
     r"breathing difficulty|gasping|choking|(lips|face|fingers) (are |is )?(turning )?(blue|grey|gray)|blue lips|cyanosis|"
     r"inhaler (isn'?t|is not|not|doesn'?t) (helping|working)|can'?t get (enough )?air"),
    ('Possible stroke signs',
     r"face (is )?droop|droop(ing|y) face|slurred speech|sudden(ly)? (confus|weak|numb)|one[- ]sided weakness|"
     r"(weakness|numbness) (on|in) one side|\b(having|had|think (i'?m|he'?s|she'?s)( having)?|signs? of|symptoms? of|possible|suspected?) (a )?stroke\b(?!\s+(last|\d+|a few|several|some|many|years?|months?|ago|before|in (19|20)\d\d))"),
    ('Loss of consciousness or seizure',
     r"unconscious|passed out|not responding|unresponsive|seizure|convuls|fit(s)? and fell|"
     r"won'?t wake|can'?t (be )?wake|hard to wake|limp and|collapsed"),
    ('Severe bleeding',
     r"bleeding (heavily|a lot|profusely|won'?t stop|that won'?t stop)|heavy bleeding|vomit(ing|ed)? blood|"
     r"cough(ing)? (up )?blood|blood in (my )?vomit|black(,)? tarry stool|severe bleeding|bleeding (is )?(not|won'?t) stop"),
    ('Severe allergic reaction',
     r"(swelling|swollen) (of |in )?(my |the )?(lips|tongue|throat|face)|(lips|tongue|throat) (are |is )?(swelling|swollen|closing)|"
     r"throat (is )?(closing|tight)|anaphyla"),
    ('Worst headache of life', r"worst headache|thunderclap|sudden severe headache"),
    ('Poisoning or overdose',
     r"overdos|swallowed (poison|bleach|pesticide|too many|a battery|a magnet|detergent)|poisoning|"
     r"took too many (pills|tablets)|drank (bleach|petrol|kerosene|antifreeze)"),
    ('Stiff neck with fever',
     r"(stiff neck|neck stiffness).{0,60}(fever|temperature)|(fever|temperature).{0,60}(stiff neck|neck stiffness)"),
    ('Fever with confusion',
     r"(confus|disorient|not making sense|incoherent).{0,60}(fever|temperature|feverish)|(fever|temperature|feverish).{0,60}(confus|disorient|not making sense)"),
    # --- additional phrasings of the same warning signs (found missing on a held-out set) ---
    ('Chest pain or pressure',
     r"clutch\w* (his|her|their|my) chest|(pain|pressure|tightness|squeez\w+|band|ache|weight) (across|around|over|on) (my |his |her |their |the )?chest|"
     r"chest.{0,25}(crushing|squeez\w+|vice|band)|crushing.{0,25}chest|chest.{0,30}(down|into|spread\w*|radiat\w*).{0,25}(jaw|left arm|arm|neck)|"
     r"(jaw|left arm).{0,30}(pain|ache).{0,30}chest|(chest|heart).{0,20}(pounding|racing).{0,40}(faint|dizzy|sweat)|grey and sweaty|gray and sweaty"),
    ('Severe trouble breathing',
     r"(isn'?t|is not|not|stopped|has stopped|have stopped|stops) breathing|no pulse|struggling for breath|fighting for breath|gasp\w* for (air|breath)|turning (blue|grey|gray)"),
    ('Possible stroke signs',
     r"(suddenly )?(can'?t|cannot|unable to) (speak|talk|get (my |her |his )?words out)|mouth (is )?(hanging|drooping|dropped|sagging)|"
     r"one side of (the |her |his |my )?(face|mouth).{0,25}(droop|hang|sag)|"
     r"(can'?t|cannot|unable to) (feel|move|lift|raise) (my |his |her )?(left |right )?(arm|leg|face)|numb(ness)? on one side|one side.{0,25}numb|face.{0,20}numb"),
    ('Loss of consciousness or seizure',
     r"jerking|shaking uncontrollably|gone stiff|going stiff|twitching all over|convulsing|isn'?t moving|not moving at all|won'?t move|lying (motionless|still)|"
     r"drowsy.{0,40}(burning up|hot).{0,40}(nonsense|confus)"),
    ('Severe allergic reaction',
     r"(tongue|lips|face|throat|eyelids) (are |is |has |have )?(puffing|puffy|puffed|ballooning)( up)?|puffing up after|"
     r"(wasp|bee|hornet) sting.{0,60}(swell|puff|breath|wheez)|wheez\w+.{0,60}(lips|face|swell)|(lips|face|swell).{0,60}wheez"),
    ('Severe bleeding',
     r"lost (a )?(huge|large|great|massive)? ?(amount|deal|lot)? ?of blood|losing (a lot of )?blood|blood (is )?(pouring|gushing|everywhere|soaking)|bleeding.{0,15}(everywhere|badly)"),
    ('Poisoning or overdose',
     r"(ate|eaten|swallowed|took|taken|drank|drunk|gulped)( a| the)? (handful|bottle|whole|lot|load|many|several|too many|too much)( of)?.{0,35}(pills?|tablets?|capsules?|medicine|medication|sleeping|paracetamol|acetaminophen|ibuprofen|aspirin)"),
    ('Fever with confusion',
     r"talking nonsense|(not|isn'?t) making (any )?sense|rambling|delirious|hallucinat\w+"),
    ('Serious injury or accident',
     r"fell (off|from|down).{0,25}(roof|ladder|tree|stairs|balcony|height|cliff)|fall(en)? off (a |the )?(roof|ladder|tree|balcony|horse|motorbike|scooter)|run over|crushed by"),
    ('Snake or scorpion bite',
     r"snake ?bite|bitten by (a |an )?(snake|cobra|viper|krait|scorpion)|scorpion sting|stung by (a )?scorpion"),
    ('Serious injury or accident',
     r"hit by (a )?(car|bus|truck|vehicle|bike)|(fell|fall|fallen) from (a )?(height|roof|tree|ladder|balcony|stairs)|"
     r"electrocuted|electric shock|drown|nearly drowned|severe head injury|stabbed|shot"),
]
CRISIS_PATTERN = re.compile(
    r"suicid|kill (myself|my self)|end (my|this) life|want to die|self[- ]harm|hurt myself|don'?t want to (live|be alive)|"
    r"better off (dead|without me)|no point (in )?(living|going on|being alive)|don'?t see the point (in )?(living|going on)|thinking (of|about) ways to (go|die|end)|"
    r"(want to|going to) disappear forever|can'?t go on|end it all|take my own life|not worth living|wish i (was|were) dead|wish i (would not|wouldn'?t) wake up")
# Urgent = needs a doctor TODAY, but not necessarily an ambulance.
URGENT_PATTERNS: List[Tuple[str, str]] = [
    ('Fever in a very young baby',
     r"(newborn|infant|\b[0-9]{1,2}\s*(day|week)s?[- ]old|\b[1-2]\s*months?[- ]old).{0,80}(fever|temperature|hot)|"
     r"(fever|temperature).{0,80}(newborn|infant|\b[0-9]{1,2}\s*(day|week)s?[- ]old|\b[1-2]\s*months?[- ]old)"),
    ('Very high fever', r"\b(10[4-9]|1[1-9][0-9])\s*(°|degrees?)?\s*f\b|\b(40|41|42)(\.\d)?\s*(°|degrees?)?\s*c\b"),
    ('Cannot keep fluids down', r"(can'?t|cannot|unable to) (keep|hold) (down )?(any )?(fluids|water|food)"),
    ('Possible appendicitis pattern',
     r"(right (lower|side).{0,25}(belly|abdomen|stomach)|lower right.{0,25}(belly|abdomen|stomach|side)).{0,60}(pain|hurts?|tender)|"
     r"(pain|hurts?).{0,40}(right (lower|side).{0,15}(belly|abdomen|stomach)|lower right)"),
    ('Pregnancy with bleeding or severe pain',
     r"pregnan\w*.{0,70}(bleed|severe (pain|cramp)|leaking fluid|reduced movement|no movement)|(bleed|severe (pain|cramp)).{0,70}pregnan"),
    ('Sudden testicular pain', r"testic\w+.{0,40}(sudden|severe|badly|swollen)|(sudden|severe).{0,30}testic\w+"),
    ('Loss of bladder or bowel control',
     r"(can'?t|cannot|lost|loss of|unable to) (control|hold).{0,15}(bladder|bowel|urine|stool)|incontinen\w+ (and|with) (back|numb|weak)"),
    ('Sudden vision loss', r"sudden(ly)? (loss of vision|vision loss|lost (my )?vision|went blind|can'?t see)"),
    ('Head injury with warning signs',
     r"(head injury|hit (my|his|her|their) head|bumped (my|his|her|their) head|banged (my|his|her) head).{0,100}(vomit|drows|confus|unconscious|blackout|blacked out|can'?t remember)"),
    ('Possible appendicitis pattern',
     r"(right (lower|side).{0,25}(tummy)|lower right.{0,25}(tummy)).{0,60}(pain|hurts?|tender)|(pain|hurts?).{0,40}(right lower tummy|lower right tummy)"),
    ('Pregnancy with bleeding or severe pain',
     r"(pregnan\w*|\d+ weeks (pregnant)?).{0,70}(stopped moving|not moving|hasn'?t moved|less movement|baby.{0,20}(isn'?t|not) moving)"),
    ('Signs of dehydration',
     r"(no|not|hasn'?t|haven'?t|without).{0,15}(wet nappy|wet nappies|wet diaper|wee|urine|pee).{0,30}(\d+ hours|all day|today|since)"),
    ('Hard belly with severe pain', r"(belly|abdomen|stomach) (is )?(rock[- ]hard|rigid|board[- ]like)"),
    ('Pregnancy warning signs',
     r"pregnan\w*.{0,80}(blurr\w* vision|severe headache|sudden swelling|swollen (face|hands))|(blurr\w* vision|severe headache).{0,80}pregnan"),
    ('Possible diabetic emergency',
     r"diabet\w*.{0,100}(vomiting|breathing fast|fast breathing|confus|fruity|very drowsy)|(vomiting|confus).{0,60}diabet"),
    ('Head injury or bleeding on blood thinners',
     r"(blood thinners?|warfarin|anticoagulant|apixaban|rivaroxaban).{0,100}(fell|hit (my )?head|head injury|bleeding)|(fell|hit (my )?head|head injury).{0,100}(blood thinners?|warfarin|anticoagulant)"),
    ('Fever with a weakened immune system',
     r"(chemo\w*|immunosuppress\w*|transplant).{0,80}(fever|temperature)|(fever|temperature).{0,80}(chemo\w*|immunosuppress\w*)"),
    ('Signs of dehydration',
     r"(barely|hardly|not|no|haven'?t|hasn'?t|without).{0,25}(passed|had)?.{0,12}(wee|urine|pee)\w*.{0,40}(all day|today|for \d+ hours|since (yesterday|morning))"),
]

# "Should be checked soon" signs: a doctor within a day or two. They raise the urgency to at least medium.
SEE_SOON_PATTERNS: List[Tuple[str, str]] = [
    ('Blood in urine, stool or sputum', r"blood in (my |the )?(urine|stool|poo|poop|sputum|phlegm)|(peeing|passing|pooing) blood|bloody (urine|stool|diarrh\w+)"),
    ('A new lump', r"(new |a )?lump (in|on|under).{0,20}(breast|testicle|neck|armpit|groin)|lump in (my |the )?breast"),
    ('Cannot bear weight after an injury', r"(can'?t|cannot|unable to) (put|bear) (any )?weight|can'?t walk on it"),
    ('Fever lasting several days', r"(fever|temperature).{0,40}(for|since|last(ed|ing)?).{0,15}(3|three|4|four|5|five|6|six|7|seven|several|few) days|(3|three|4|four|5|five|6|six|7|seven|several|few) days (of|with) (a )?(high )?(fever|temperature)"),
    ('Unexplained weight loss', r"(unexplained|unintentional) weight loss|lost (a lot of )?weight (without|for no)"),
    ('Severe eye pain', r"(stabbing|severe|sudden|intense) (pain|ache) (behind|in) (my |the |her |his )?eyes?"),
]
_SEE_SOON_RE = [(label, re.compile(p)) for label, p in SEE_SOON_PATTERNS]
_EMERGENCY_RE = [(label, re.compile(p)) for label, p in EMERGENCY_PATTERNS]
_URGENT_RE = [(label, re.compile(p)) for label, p in URGENT_PATTERNS]


@dataclass
class Signals:
    emergency: List[str] = field(default_factory=list)
    urgent: List[str] = field(default_factory=list)
    soon: List[str] = field(default_factory=list)
    crisis: bool = False

    @property
    def any_emergency(self) -> bool:
        return bool(self.emergency or self.crisis)

    def as_dict(self) -> Dict:
        return {'emergency': self.emergency, 'urgent': self.urgent, 'crisis': self.crisis}   # (the 'soon' tier is internal)


def scan(text: Optional[str]) -> Signals:
    """Scan free text for emergency / urgent / self-harm language. Safe on any input."""
    t = (text or '').lower()
    sig = Signals()
    for label, rx in _EMERGENCY_RE:
        for m in rx.finditer(t):
            if not negated(t, m.start()):
                if label not in sig.emergency:
                    sig.emergency.append(label)
                break
    for label, rx in _URGENT_RE:
        for m in rx.finditer(t):
            if not negated(t, m.start()):
                if label not in sig.urgent:
                    sig.urgent.append(label)
                break
    for label, rx in _SEE_SOON_RE:
        for m in rx.finditer(t):
            if not negated(t, m.start()):
                sig.soon.append(label)
                break
    sig.crisis = bool(CRISIS_PATTERN.search(t))
    # Hindi / Marathi / Kannada, in their own scripts and typed in English letters. Same labels, so everything downstream is unchanged.
    extra, extra_crisis = i18n_emergency.detect(text or '')
    for label in extra:
        if label not in sig.emergency:
            sig.emergency.append(label)
    sig.crisis = sig.crisis or extra_crisis
    return sig


def scan_red_flags(raw_text: Optional[str]) -> Dict:
    """Dict form of scan(), kept for older callers."""
    return scan(raw_text).as_dict()


# ---------------------------------------------------------------------------
# Emergency guidance
# ---------------------------------------------------------------------------

CRISIS_SUPPORT = {
    'IN': 'You are not alone, and help is available right now. Call Tele-MANAS (free, 24/7) on 14416, or {EMERGENCY_NUMBER} if you are in immediate danger.',
    'DEFAULT': 'You are not alone, and help is available right now. Call {EMERGENCY_NUMBER} if you are in immediate danger, or contact a local crisis line.',
}

GUIDANCE: Dict[str, List[str]] = {
    'Chest pain or pressure': [
        f'Call {N} now and say you have chest pain.',
        'Stop what you are doing, sit down and stay as calm and still as you can.',
        'Loosen tight clothing. If you are alone, unlock the door so help can reach you.',
        'Do not drive yourself. Follow the instructions the emergency operator gives you.'],
    'Severe trouble breathing': [
        f'Call {N} now.',
        'Sit upright, leaning slightly forward, and loosen tight clothing.',
        'Use your prescribed reliever inhaler if you have one, exactly as directed.',
        'Stay with the person. If they stop breathing, the operator will guide you through CPR.'],
    'Possible stroke signs': [
        f'Call {N} now and say "possible stroke".',
        'Note the exact time the symptoms started and tell the operator.',
        'Do not give food, drink or medicine.',
        'Keep the person comfortable with the head slightly raised, and stay with them.'],
    'Loss of consciousness or seizure': [
        f'Call {N} now.',
        'During a seizure: clear space around them, cushion the head, and never put anything in the mouth or hold them down.',
        'When it stops, gently roll them onto their side.',
        'If they are unresponsive and not breathing normally, the operator will guide you through CPR.'],
    'Severe bleeding': [
        f'Call {N} now.',
        'Press firmly on the wound with a clean cloth and keep pressing. Do not lift it to check.',
        'If blood soaks through, add more cloth on top rather than removing it.',
        'Lie the person down and keep them warm until help arrives.'],
    'Severe allergic reaction': [
        f'Call {N} now.',
        'If they have an adrenaline auto-injector, use it as instructed on the device.',
        'Lie them flat with legs raised, or sitting up if breathing is hard. Do not let them stand.',
        "Stay with them and follow the operator's instructions."],
    'Worst headache of life': [
        f'Call {N} now.',
        'Lie down somewhere quiet and do not take extra medicine until you have been assessed.',
        'Do not be left alone. Tell the operator if there was a head injury or any weakness or confusion.'],
    'Poisoning or overdose': [
        f'Call {N} now (or your local poison helpline if you have the number).',
        'Keep the container, packet or a photo of it to show the responders.',
        'Do not try to make them vomit unless a professional tells you to.',
        'If they are drowsy or unresponsive, keep them on their side and stay with them.'],
    'Stiff neck with fever': [
        f'Call {N} or go to the nearest emergency department now. Do not wait to see if it passes.',
        'Do not drive yourself if you feel confused or drowsy.',
        'Take a list of any medicines you use.'],
    'Fever with confusion': [
        f'Call {N} now. Fever with confusion can mean a serious infection.',
        'Keep the person safe and do not leave them alone.',
        'Do not give anything to eat or drink if they are very drowsy.'],
    'Snake or scorpion bite': [
        f'Call {N} now and go to the nearest hospital. Do not wait for symptoms.',
        'Keep the person calm and as still as possible, with the bitten limb resting at or below heart level.',
        'Remove rings, watches and tight clothing near the bite. Do not cut the wound, suck out venom or apply a tight tourniquet.',
        'Note the time and what the animal looked like, but do not try to catch it.'],
    'Serious injury or accident': [
        f'Call {N} now.',
        'Do not move the person unless they are in immediate danger (for example fire or traffic).',
        'Control any heavy bleeding with firm pressure and keep them warm until help arrives.'],
}
DEFAULT_STEPS = [f'Call {N} now.', 'Stay with the person and keep them calm and comfortable.',
                 'Follow the instructions the emergency operator gives you.']
EMERGENCY_MESSAGE = 'EMERGENCY SYMPTOMS DETECTED - SEEK IMMEDIATE MEDICAL ATTENTION'
EMERGENCY_ACTION = f'Call {N} or go to the nearest emergency room immediately'


def _lang() -> str:
    return i18n.current()


def emergency_message(lang: Optional[str] = None) -> str:
    tx = i18n_emergency.TEXT.get(lang or _lang())
    return tx['message'] if tx else EMERGENCY_MESSAGE


def emergency_action(region: str, lang: Optional[str] = None) -> str:
    tx = i18n_emergency.TEXT.get(lang or _lang())
    return localize_text(tx['action'] if tx else EMERGENCY_ACTION, region)


def crisis_support_text(region: str, lang: Optional[str] = None) -> str:
    lang = lang or _lang()
    if lang in i18n_emergency.LANGS:
        table = i18n_emergency.CRISIS_SUPPORT_IN if region == 'IN' else i18n_emergency.CRISIS_SUPPORT_OTHER
        return localize_text(table[lang], region)
    return localize_text(CRISIS_SUPPORT.get(region, CRISIS_SUPPORT['DEFAULT']), region)


def label_for(label: str, lang: Optional[str] = None) -> str:
    """The warning-sign name in the person's language (English labels stay the internal keys)."""
    return i18n_emergency.LABELS.get(lang or _lang(), {}).get(label, label)


def build_emergency(reasons: List[str], region: str, crisis: bool = False, lang: Optional[str] = None) -> Dict:
    """Localised emergency guidance: {'reasons', 'headline', 'steps', 'crisis_support'} in the current language (English by default)."""
    lang = lang or _lang()
    translated = lang in i18n_emergency.LANGS
    steps: List[str] = []
    opener_added = False
    for label in reasons:
        label_steps = (i18n_emergency.GUIDANCE[lang].get(label) if translated else None) or GUIDANCE.get(label, [])
        for i, step in enumerate(label_steps):
            # Several warning signs share the same "Call N now" opener: say it once.
            if step in steps or (i == 0 and opener_added):
                continue
            if i == 0:
                opener_added = True
            steps.append(step)
    if not steps and not crisis:
        steps = list(i18n_emergency.DEFAULT_STEPS[lang] if translated else DEFAULT_STEPS)
    support = None
    if crisis:
        support = crisis_support_text(region, lang)
        if not reasons:
            steps = list(i18n_emergency.CRISIS_STEPS[lang]) if translated else [
                f'Call {N} if you are in immediate danger.', 'Stay with someone you trust, or ask someone to stay with you.',
                'Move anything you could use to hurt yourself out of reach.']
    shown = list(reasons) + (['Thoughts of self-harm'] if crisis and not reasons else [])
    return {'reasons': [label_for(r, lang) for r in shown] if translated else shown,
            'headline': i18n_emergency.TEXT[lang]['headline'] if translated else 'This could be an emergency',
            'steps': [localize_text(s, region) for s in steps], 'crisis_support': support}


def emergency_analysis(reasons: List[str], region: str, crisis: bool, summary: str, kind: str = 'symptom') -> Dict:
    """Immediate-guidance result shaped like a normal analysis, so it renders and saves like one."""
    lang = _lang()
    tx = i18n_emergency.TEXT.get(lang)
    info = build_emergency(reasons, region, crisis, lang)
    recs = list(info['steps'])
    if info['crisis_support']:
        recs.insert(0, info['crisis_support'])
    result = {
        'summary': tx['summary'] if tx else summary, 'detected_symptoms': [], 'possible_conditions': [], 'possible_explanations': [],
        'reported': {'symptoms': [], 'details': []},
        'urgency_level': 'high', 'care_level': 'emergency', 'timeframe': tx['timeframe'] if tx else 'Right now',
        'emergency_alert': {'alert': True, 'message': emergency_message(lang), 'action': emergency_action(region, lang),
                            'reason': '; '.join(info['reasons'])},
        'red_flags_present': info['reasons'], 'red_flags': info['reasons'],
        'recommendations': recs, 'avoid': list(tx['avoid']) if tx else ['Do not wait to see whether it gets better', 'Do not drive yourself'],
        'see_doctor_when': [], 'information_gaps': [], 'safety_tips': [],
        'analysis_mode': 'rules', 'assessment_status': 'complete', 'interview_stopped_early': True,
        'disclaimer': tx['disclaimer'] if tx else 'Emergency guidance only. It is not a diagnosis. Call emergency services.',
    }
    result['reliability'] = {'level': 'good', 'label': 'Act first', 'reasons': ['A warning sign was reported, so guidance is given immediately without further questions.'],
                             'note': reliability.NOTE}
    result['uncertainty_statement'] = ('We cannot tell what is causing this from here. Because a warning sign was mentioned, '
                                       'the safe choice is to get emergency help now.')
    result['safety_review'] = {'engine_version': ENGINE_VERSION, 'checks': ['emergency_scan'], 'adjustments': [],
                               'original_urgency': 'high', 'final_urgency': 'high', 'urgency_raised': False,
                               'language_softened': False, 'removed': []}
    return result


def immediate_emergency(text: str, region: str, summary: str) -> Optional[Dict]:
    """If the text contains an emergency or self-harm signal, return ready-made guidance; else None."""
    sig = scan(text)
    if sig.any_emergency:
        result = emergency_analysis(sig.emergency, region, sig.crisis, summary)
        from quickaid.ai import first_aid                    # a specific guide (choking) outranks the generic guidance for the label it overlaps with
        emergency_guides = [g for g in first_aid.match(text) if g.care_for(text)[0] == 'emergency']
        if emergency_guides and _lang() == 'en':
            first_aid.apply(result, emergency_guides, text, lambda s: localize_text(s, region))
            result.pop('_first_aid_count', None)
            result.pop('_replace_generic_emergency_steps', None)
            result['summary'] = summary or result['summary']
        return result
    return None


# ---------------------------------------------------------------------------
# 2. Floors: rules that set a MINIMUM urgency
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Floor:
    level: str
    reason: str
    source: str            # text | context | model_output | image
    emergency: bool = False


# Serious conditions: if the AI itself names one as a leading possibility, the person must not be told "low".
SERIOUS_HIGH = ['heart attack', 'myocardial infarction', 'stroke', 'meningitis', 'sepsis', 'septic', 'pulmonary embolism', 'anaphylaxis',
                'ectopic pregnancy', 'aortic', 'subarachnoid', 'brain haemorrhage', 'brain hemorrhage', 'diabetic ketoacidosis',
                'pneumothorax', 'testicular torsion', 'cauda equina', 'epiglottitis', 'pre-eclampsia', 'preeclampsia',
                'overdose', 'carbon monoxide', 'toxic ingestion', 'bowel obstruction', 'peritonitis', 'encephalitis']
SERIOUS_MEDIUM = ['pneumonia', 'appendicitis', 'kidney infection', 'pyelonephritis', 'cellulitis', 'dehydration', 'fracture',
                  'concussion', 'urinary tract infection', 'gallstone', 'cholecystitis', 'abscess', 'deep vein thrombosis', 'dvt',
                  'angina', 'heart problems', 'anaemia', 'anemia']


def _contains(text: str, terms: Iterable[str]) -> Optional[str]:
    t = text.lower()
    for term in terms:
        if re.search(r'\b' + re.escape(term), t):
            return term
    return None


def text_floors(sig: Signals) -> List[Floor]:
    floors = [Floor('high', f'Warning sign reported: {lbl}', 'text', True) for lbl in sig.emergency]
    if sig.crisis:
        floors.append(Floor('high', 'Thoughts of self-harm were mentioned', 'text', True))
    floors += [Floor('high', f'Reported: {lbl}', 'text') for lbl in sig.urgent]
    floors += [Floor('medium', f'Reported: {lbl}', 'text') for lbl in sig.soon]
    return floors


def context_floors(ctx: Dict, symptoms: Iterable[str], text: str) -> List[Floor]:
    """Age, history, severity, duration and trend. Can only raise urgency."""
    syms = set(symptoms) | set(ctx.get('detected') or [])
    t = text.lower()
    fever = 'fever' in syms or 'fever' in t
    belly = 'abdominal_pain' in syms or 'abdominal' in t or 'stomach' in t
    breathing = 'shortness_of_breath' in syms or 'breath' in t
    hist = set(ctx.get('history') or [])
    sev = ctx.get('severity') or 0
    dur = DURATION_RANK.get(ctx.get('duration'), 0)
    age = ctx.get('age_group')
    out: List[Floor] = []

    def add(level, why):
        out.append(Floor(level, why, 'context'))

    if ctx.get('urgent_flags'):
        add('high', 'You reported: ' + '; '.join(ctx['urgent_flags'][:3]))
    if sev >= 9:
        add('high', f'Severity is {sev} out of 10')
    elif sev >= 7:
        add('medium', f'Severity is {sev} out of 10')
    if ctx.get('course') == 'worse':
        add('high' if sev >= 7 else 'medium', 'It is getting worse')
    if dur >= 4:
        add('medium', 'It has lasted more than 2 weeks')
    elif dur >= 3 and fever:
        add('medium', 'A fever lasting more than a week')
    elif dur >= 2 and fever and age in ('infant', 'child'):
        add('medium', 'Fever lasting several days in a child')
    if age == 'infant' and fever:
        add('high', 'Fever in a baby under 1 year')
    if age == 'infant' and (belly or breathing or 'vomiting' in syms):
        add('medium', 'Illness in a baby can change quickly')
    if age == 'older' and (fever or breathing or belly):
        add('medium', 'Older adults can become seriously ill faster')
    if 'immune' in hist and fever:
        add('high', 'Fever with a weakened immune system')
    if 'pregnant' in hist and (fever or belly or 'bleeding' in t):
        add('high', 'Possible pregnancy with fever or belly pain')
    if hist & {'heart', 'lung', 'diabetes', 'kidney_liver'} and (fever or breathing or belly or dur >= 2):
        add('medium', 'A health condition raises the risk')
    if 'blood_thinners' in hist and any(w in t for w in ('fall', 'bump', 'head', 'bleeding')):
        add('high', 'Injury or bleeding while on blood thinners')
    return out


def symptom_output_floors(result: Dict) -> List[Floor]:
    """Look at what the MODEL said, and refuse to let 'low' stand next to a serious possibility."""
    floors: List[Floor] = []
    expl = result.get('possible_explanations') or []
    names = [(e.get('name') or '', e.get('likelihood')) for e in expl] or [(c, None) for c in result.get('possible_conditions') or []]
    for i, (name, likelihood) in enumerate(names):
        leading = i == 0 or likelihood == 'more likely'
        hit = _contains(name, SERIOUS_HIGH)
        if hit:
            floors.append(Floor('high' if leading else 'medium',
                                f'"{name}" was listed as a {"leading " if leading else ""}possibility', 'model_output'))
            continue
        hit = _contains(name, SERIOUS_MEDIUM)
        if hit and leading:
            floors.append(Floor('medium', f'"{name}" was listed as a leading possibility', 'model_output'))
    if result.get('red_flags_present'):
        floors.append(Floor('medium', 'A warning sign is already present', 'model_output'))
        sig = scan(' . '.join(result['red_flags_present']))
        floors += text_floors(sig)
    return floors


# Image text patterns -> floors. Emergency: call now. Same-day: doctor today. Medium: doctor soon.
IMAGE_EMERGENCY = re.compile(
    r"spurting|arterial|bleeding (heavily|profusely|a lot|that won'?t stop|won'?t stop)|(exposed|visible|protruding) (bone|tendon|muscle|fat)|"
    r"bone (is )?(visible|protruding|showing)|amputat|charred|full[- ]thickness|third[- ]degree|burn.{0,30}(airway|inhal)|"
    r"swelling of (the )?(lips|tongue|throat|face)|snake ?bite|severe(ly)? deform")
IMAGE_SAME_DAY = re.compile(
    r"red streak|spreading (redness|infection)|expanding redness|pus.{0,30}swell|necrotic|necrosis|black(ened)? (skin|tissue)|dusky|"
    r"(large|extensive|deep) burn|burn.{0,40}(face|hand|hands|foot|feet|genital|joint)|(animal|dog|cat|human|monkey) bite|"
    r"embedded (object|glass|splinter|metal)|deformity|deformed|angulated|compound fracture|gaping|deep (wound|cut|laceration|gash)|"
    r"(limb|hand|foot|finger).{0,20}(pale|blue|mottled|cold)")
IMAGE_MEDIUM = re.compile(
    r"stitches|sutur|blister|second[- ]degree|possible fracture|suspected fracture|sprain|significant swelling|infection|infected|\bpus\b|cellulitis|abscess|"
    r"spreading rash|deep(er)? (laceration|puncture)|ulcer")


def image_floors(result: Dict) -> List[Floor]:
    texts = ([f.get('description', '') for f in result.get('findings') or []] + (result.get('red_flags_present') or [])
             + [e.get('name', '') for e in result.get('possible_explanations') or []] + (result.get('observations') or [])
             + (result.get('detected_conditions') or []) + [result.get('summary') or ''])
    floors: List[Floor] = []
    for raw in texts:
        t = (raw or '').lower()
        if not t:
            continue
        # skip statements that clearly deny the sign ("no exposed bone")
        def hit(rx):
            for m in rx.finditer(t):
                if not negated(t, m.start()):
                    return m.group(0)
            return None
        if h := hit(IMAGE_EMERGENCY):
            floors.append(Floor('high', f'The image analysis mentions "{h}"', 'image', True))
        elif h := hit(IMAGE_SAME_DAY):
            floors.append(Floor('high', f'The image analysis mentions "{h}"', 'image'))
        elif h := hit(IMAGE_MEDIUM):
            floors.append(Floor('medium', f'The image analysis mentions "{h}"', 'image'))
    if result.get('red_flags_present'):
        floors.append(Floor('medium', 'A warning sign is visible now', 'image'))
    return floors


# ---------------------------------------------------------------------------
# 3. Language and content safety
# ---------------------------------------------------------------------------

_DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|µg|g|ml|iu|units?|tablets?|pills?|capsules?|drops?|puffs?)\b", re.I)
_STOP_MED_RE = re.compile(r"\b(stop|discontinue|quit|skip)\b.{0,25}\b(taking|using)?\b.{0,20}\b(your )?(medic\w+|tablets?|pills?|insulin|prescription)\b", re.I)
DOSE_REPLACEMENT = 'Ask a pharmacist or doctor which medicine is right for you and how much to take.'
STOP_REPLACEMENT = 'Do not stop prescribed medicine without asking your doctor.'

# (pattern, replacement) - applied to AI-written text only, never to our own disclaimers.
# Rewrites of CLAIMS ABOUT WHAT THE PERSON HAS. These only make sense in summaries and explanations: in advice lists, "You have a
# fever" is a trigger ("see a doctor if you have a fever"), not a diagnosis, so it must be left alone there.
_DIAGNOSTIC: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\byou (?:definitely |certainly |clearly |obviously )?have (an? |the )", re.I), r"you may have \1"),
    (re.compile(r"\bthis is (?:definitely |certainly |clearly |obviously )(an? |the )", re.I), r"this could be \1"),
    (re.compile(r"\b(?:you were |you are |is |was )?diagnosed with\b", re.I), "may have"),
    (re.compile(r"\b(?:confirmed|confirms|confirm) (?:a |the )?diagnosis\b", re.I), "is not a diagnosis"),
    (re.compile(r"\bdiagnosis:\s*", re.I), "Possible explanation: "),
]
# Overconfidence and false reassurance: safe to rewrite anywhere.
_OVERCONFIDENT: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:definitely|certainly|undoubtedly|unquestionably|without (?:a )?doubt|100% ?(?:sure|certain)|guaranteed)\b", re.I), "possibly"),
    (re.compile(r"\b(?:nothing to worry about|no need to worry|no cause for concern|perfectly (?:fine|safe)|completely (?:safe|harmless)|totally harmless)\b", re.I),
     "likely minor, though that cannot be confirmed from here"),
    (re.compile(r"\bno need to (?:see|visit|consult) (?:a |the )?(?:doctor|clinician|professional)\b", re.I), "you may still want to check with a doctor"),
    (re.compile(r"\b(?:safe|fine) to (?:ignore|wait|leave it)\b", re.I), "worth keeping an eye on"),
]
_REASSURANCE_RE = re.compile(
    r"nothing to worry|no need to worry|no cause for concern|no need to (see|visit|consult)|safe to (ignore|wait)|"
    r"will (go away|clear up|resolve) (on its own|by itself)|perfectly (fine|safe)|completely (safe|harmless)", re.I)


def _keep_case(rep: str):
    """Replacement function that keeps a leading capital letter, so rewritten sentences still read properly."""
    def sub(m: re.Match) -> str:
        out = m.expand(rep)
        return out[:1].upper() + out[1:] if m.group(0)[:1].isupper() else out
    return sub


def soften(text: str, diagnostic: bool = True) -> Tuple[str, bool]:
    """Rewrite overconfident or falsely reassuring wording in AI text. Returns (text, changed).

    `diagnostic=False` skips the "you have X" style rewrites (used for advice lists, where such phrases are triggers).
    """
    new = text
    for rx, rep in (_DIAGNOSTIC if diagnostic else []) + _OVERCONFIDENT:
        new = rx.sub(_keep_case(rep), new)
    phrase = re.escape("likely minor, though that cannot be confirmed from here")
    new = re.sub(rf"({phrase})(?:[,.]?\s*(?:and\s+)?{phrase})+", r"\1", new)
    return new, new != text


def _scrub_lines(lines: List[str], removed: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        if _DOSE_RE.search(line):
            removed.append('medicine dose')
            line = DOSE_REPLACEMENT
        elif _STOP_MED_RE.search(line) and 'do not stop' not in line.lower() and "don't stop" not in line.lower():
            removed.append('advice to stop medicine')
            line = STOP_REPLACEMENT
        if line not in out:
            out.append(line)
    return out


_TEXT_FIELDS = ('summary', 'quality_note')
_LIST_FIELDS = ('recommendations', 'avoid', 'see_doctor_when', 'red_flags_present', 'red_flags', 'warning_signs',
                'information_gaps', 'safety_tips', 'observations', 'not_assessable', 'urgency_reasons')


def sanitize_language(r: Dict) -> Dict:
    """Soften overconfident wording and strip unsafe medicine advice in place. Returns what changed."""
    softened = False
    removed: List[str] = []
    for f in _TEXT_FIELDS:
        if isinstance(r.get(f), str) and r[f]:
            r[f], ch = soften(r[f])
            softened |= ch
    for f in _LIST_FIELDS:
        if isinstance(r.get(f), list):
            new = []
            for item in r[f]:
                s, ch = soften(str(item), diagnostic=False)
                softened |= ch
                new.append(s)
            r[f] = new
    for f in ('recommendations', 'avoid', 'see_doctor_when', 'safety_tips'):
        if isinstance(r.get(f), list):
            r[f] = _scrub_lines(r[f], removed)
    for key in ('possible_explanations',):
        for e in r.get(key) or []:
            for sub in ('name', 'why'):
                if e.get(sub):
                    e[sub], ch = soften(e[sub])
                    softened |= ch
            e['supported_by'] = [soften(str(x))[0] for x in e.get('supported_by') or []]
    for fnd in r.get('findings') or []:
        if fnd.get('description'):
            fnd['description'], ch = soften(fnd['description'])
            softened |= ch
    if r.get('possible_explanations'):
        r['possible_conditions' if 'possible_conditions' in r else 'detected_conditions'] = [e['name'] for e in r['possible_explanations']]
    return {'softened': softened, 'removed': sorted(set(removed))}


# ---------------------------------------------------------------------------
# 4. The engine
# ---------------------------------------------------------------------------

def _max_level(a: str, b: str) -> str:
    return a if RANK.get(a, 2) >= RANK.get(b, 2) else b


class TriageEngine:
    """Stateless; one shared instance is exposed as `engine` at the bottom of this module."""

    # -- decisions ---------------------------------------------------------

    @staticmethod
    def decide(model_level: str, floors: List[Floor], model_emergency: bool = False) -> Dict:
        """Combine the model's level with every floor. Level can only go up."""
        level = model_level if model_level in RANK else 'medium'
        original = level
        reasons: List[str] = []
        emergency = bool(model_emergency)
        for fl in floors:
            if RANK[fl.level] > RANK[level]:
                level = fl.level
            if RANK[fl.level] >= RANK[original] and fl.reason not in reasons:
                reasons.append(fl.reason)
            emergency = emergency or fl.emergency
        if emergency:
            level = 'high'
        care = 'emergency' if emergency else CARE_FOR_LEVEL[level]
        return {'level': level, 'original': original, 'care_level': care, 'emergency': emergency,
                'reasons': reasons, 'raised': RANK[level] > RANK[original]}

    @staticmethod
    def timeframe_for(care: str, model_timeframe: str, ctx: Optional[Dict] = None) -> str:
        ctx = ctx or {}
        mt = (model_timeframe or '').strip()
        if care == 'emergency':
            return 'Right now'
        if care == 'same_day':
            return 'Today'
        if care == 'soon':
            if mt and not re.search(r'home|self[- ]care|no (need|treatment)', mt, re.I):
                return mt
            return 'Within 24 hours' if ctx.get('course') == 'worse' else 'Within 1–2 days'
        if mt and re.search(r'home|self|see a doctor if|monitor', mt, re.I):
            return mt
        return ('Home care; see a doctor if no better in 3–5 days'
                if DURATION_RANK.get(ctx.get('duration'), 0) < 3 else 'Home care; see a doctor if it is not improving')

    # -- shaping -----------------------------------------------------------

    @staticmethod
    def _shape_symptom(r: Dict) -> None:
        r.setdefault('summary', '')
        for k in ('detected_symptoms', 'possible_conditions', 'possible_explanations', 'red_flags_present', 'red_flags',
                  'recommendations', 'avoid', 'see_doctor_when', 'information_gaps', 'safety_tips'):
            if not isinstance(r.get(k), list):
                r[k] = []
        if not r['detected_symptoms'] and (r.get('reported') or {}).get('symptoms'):
            r['detected_symptoms'] = list(r['reported']['symptoms'])
        if not r['possible_explanations'] and r['possible_conditions']:
            r['possible_explanations'] = [{'name': c, 'likelihood': 'more likely' if i == 0 else 'possible', 'why': '', 'supported_by': []}
                                          for i, c in enumerate(r['possible_conditions'][:4])]
        for e in r['possible_explanations']:
            e.setdefault('why', '')
            e.setdefault('supported_by', [])
        r['possible_conditions'] = [e['name'] for e in r['possible_explanations']]
        if not isinstance(r.get('emergency_alert'), dict):
            r['emergency_alert'] = {'alert': bool(r.get('emergency_alert')), 'message': '', 'action': '', 'reason': ''}
        r['emergency_alert'].setdefault('reason', '')
        r.setdefault('reported', {'symptoms': list(r['detected_symptoms']), 'details': []})
        if r.get('urgency_level') not in RANK:
            r['urgency_level'] = 'medium'

    @staticmethod
    def _reported_details(ctx: Optional[Dict]) -> List[Dict]:
        if not ctx:
            return []
        out = []
        if ctx.get('age_group'):
            out.append({'label': 'Who it is for', 'value': AGE_LABEL.get(ctx['age_group'], ctx['age_group'])})
        if ctx.get('duration'):
            out.append({'label': 'How long', 'value': DURATION_LABEL.get(ctx['duration'], ctx['duration'])})
        if ctx.get('severity'):
            out.append({'label': 'Severity', 'value': f"{ctx['severity']} out of 10"})
        if ctx.get('course'):
            out.append({'label': 'Trend', 'value': {'better': 'Getting better', 'same': 'About the same', 'worse': 'Getting worse',
                                                    'comes_goes': 'Comes and goes'}.get(ctx['course'], ctx['course'])})
        if ctx.get('history'):
            out.append({'label': 'Medical history', 'value': ', '.join(ctx['history'])})
        if ctx.get('selected'):
            out.append({'label': 'Also reported', 'value': ', '.join(ctx['selected'][:8])})
        return out

    # -- symptom review ----------------------------------------------------

    def review_symptom(self, result: Dict, *, text: str, region: str, mode: str, ctx: Optional[Dict] = None,
                       interview: bool = False, answered: Optional[int] = None, skipped: Optional[int] = None,
                       extra_text: Iterable[str] = ()) -> Dict:
        """Validate and finalize a symptom assessment before it is displayed or saved.

        `text` is the person's first message; `extra_text` holds free-text answers.
        Returns a NEW dict. Safe to call twice (it only ever raises urgency).
        """
        r = copy.deepcopy(result)
        first_aid_count = int(r.pop('_first_aid_count', 0) or 0)
        replace_generic = bool(r.pop('_replace_generic_emergency_steps', False))
        self._shape_symptom(r)
        checks = ['schema_shape', 'emergency_scan', 'model_output_scan', 'serious_condition_check', 'language_check', 'medication_advice_check']
        original = r['urgency_level']
        model_emergency = bool(r['emergency_alert'].get('alert'))
        adjustments: List[str] = []

        had_reassurance = bool(_REASSURANCE_RE.search(r.get('summary', '') or ''))   # before softening rewrites it
        lang = sanitize_language(r)

        sig = scan(' . '.join([text or ''] + list(extra_text)))
        floors = text_floors(sig) + symptom_output_floors(r)
        if ctx:
            floors += context_floors(ctx, r['detected_symptoms'], text or '')
            checks.append('risk_rules')
        d = self.decide(original, floors, model_emergency)
        final, care = d['level'], d['care_level']

        if d['raised']:
            adjustments.append(f"Urgency raised from {original} to {final} by safety rules: " + '; '.join(d['reasons'][:4]) + '.')
        r['urgency_level'] = final
        r['care_level'] = care
        r['timeframe'] = self.timeframe_for(care, r.get('timeframe', ''), ctx)
        r['urgency_reasons'] = list(dict.fromkeys((r.get('urgency_reasons') or []) + d['reasons']))

        # emergency block + guidance
        alert = r['emergency_alert']
        if d['emergency']:
            reasons = sig.emergency or [f.reason for f in floors if f.emergency][:2] or ([alert['reason']] if alert.get('reason') else [])
            labels = [x for x in reasons if x in GUIDANCE]
            info = build_emergency(labels or [], region, crisis=sig.crisis)
            alert.update({'alert': True, 'message': emergency_message(), 'action': emergency_action(region)})
            alert['reason'] = alert.get('reason') or '; '.join(reasons or info['reasons'])
            # Show WHICH warning sign the safety rules found, in the person's language, even when the model did not list it itself.
            r['red_flags_present'] = list(dict.fromkeys(list(r.get('red_flags_present') or []) + [label_for(x) for x in reasons]))
            generic = info['steps']
            if replace_generic:
                # A specific first-aid guide (choking) outranks the generic guidance for the label it overlaps with. In English the guide's own first step is
                # the call line; in other languages keep the translated call line, then the guide's steps.
                generic = [] if i18n.current() == 'en' else info['steps'][:1]
            steps = ([info['crisis_support']] if info['crisis_support'] else []) + generic
            r['recommendations'] = [s for s in steps if s] + [x for x in r['recommendations'] if x not in steps]
            adjustments.append('Emergency guidance added because a warning sign was detected.') if not model_emergency else None
        else:
            alert.update({'alert': False, 'message': '', 'action': '', 'reason': ''})
            # the original advice may talk about the ER while the engine's care level is "see a doctor": keep them consistent
            if mode in ('basic', 'rules'):
                self._align_basic_recommendations(r, care, r['timeframe'], region, first_aid_count)

        if sig.crisis:
            support = crisis_support_text(region)
            if support not in r['recommendations']:
                r['recommendations'].insert(0, support)

        # reassurance that contradicts the final urgency
        if RANK[final] >= RANK['medium'] and had_reassurance:
            adjustments.append('Reassuring wording was removed because the result needs medical attention.')

        # reported details
        r['reported']['symptoms'] = list(r['detected_symptoms'])
        details = self._reported_details(ctx)
        if details:
            r['reported']['details'] = details
        if not r['recommendations']:
            r['recommendations'] = ['Monitor your symptoms', 'Rest and drink fluids', 'See a doctor if symptoms get worse or you are unsure']
        if not r['red_flags'] and r.get('warning_signs'):
            r['red_flags'] = list(r['warning_signs'])

        # information gaps: say what we could not assess
        gaps = r['information_gaps']
        if ctx is not None:
            if not ctx.get('age_group'):
                gaps.append('Age was not given, which can change how urgent symptoms are.')
            if not ctx.get('duration'):
                gaps.append('How long this has lasted was not given.')
            if ctx.get('history') is None:
                gaps.append('Medical history and medicines were not covered.')
        elif not interview:
            gaps.append('Age, how long this has lasted and medical history were not given.')
        r['information_gaps'] = list(dict.fromkeys(gaps))

        # status
        no_info = not r['detected_symptoms'] and not r['possible_explanations'] and not d['emergency']
        r['assessment_status'] = 'insufficient' if no_info else ('limited' if mode in ('basic', 'rules') or not r['possible_explanations'] else 'complete')

        rel = reliability.assess('symptom', mode=mode, ai_self_rating=None, answered=answered, skipped=skipped, interview=interview,
                                 input_chars=len(text or ''), symptoms_found=len(r['detected_symptoms']),
                                 safety_overrode=d['raised'] and mode == 'ai', insufficient=no_info,
                                 explanations=len(r['possible_explanations']))
        r['reliability'] = rel
        r['uncertainty_statement'] = reliability.uncertainty_statement(rel['level'])
        self._finish(r, mode)

        if lang['softened']:
            adjustments.append('Overconfident wording was softened so the result is not presented as a diagnosis.')
        for kind in lang['removed']:
            adjustments.append(f'Removed {kind} from the advice (ask a pharmacist or doctor instead).')
        r['safety_review'] = {'engine_version': ENGINE_VERSION, 'checks': checks, 'adjustments': adjustments,
                              'original_urgency': original, 'final_urgency': final, 'urgency_raised': d['raised'],
                              'language_softened': lang['softened'], 'removed': lang['removed']}
        return r

    @staticmethod
    def _align_basic_recommendations(r: Dict, care: str, tf: str, region: str, first_aid_count: int = 0) -> None:
        """Rule-based advice was written per symptom; make it agree with the final care level."""
        n = localize_text(N, region)
        drop = ('See a doctor within 24 hours', 'Seek care sooner if symptoms get worse', 'SEEK IMMEDIATE', 'Call ', 'Do not delay care')
        kept = [x for x in r['recommendations'] if not x.startswith(drop) and 'emergency room now' not in x]
        lead = {'soon': [f'See a doctor: {tf[0].lower() + tf[1:]}', 'Go sooner if it gets worse'],
                'same_day': ['Get seen by a doctor today', f'Go to the emergency department or call {n} if it gets worse or you feel unsafe'],
                'self_care': []}.get(care, [])
        rest = [x for x in kept if x not in lead]
        if care in ('soon', 'self_care'):
            # Practical first aid / self-care comes FIRST; "see a doctor" and its timing follow it (the banner above already states the urgency).
            head = rest[:5]
            r['recommendations'] = head + lead + rest[5:]
        elif care == 'same_day' and first_aid_count:
            # "Get seen today" is stated in the banner; the time-critical first-aid steps (e.g. wash a bite for 15 minutes) go first, then the instruction.
            r['recommendations'] = rest[:3] + lead + rest[3:]
        else:
            r['recommendations'] = lead + rest              # urgent: the instruction to get seen leads

    @staticmethod
    def _finish(r: Dict, mode: str) -> None:
        disc = r.get('disclaimer') or ''
        if 'not a diagnosis' not in disc.lower():
            disc = (disc + ' ' if disc else '') + (DISCLAIMER_AI if mode == 'ai' else DISCLAIMER_RULES)
        r['disclaimer'] = disc.strip()
        r['analysis_mode'] = 'ai' if mode == 'ai' else ('rules' if mode == 'rules' else 'basic')
        r['interpretation_status'] = 'unvalidated_ai' if mode == 'ai' else 'rules'
        r['interpretation_note'] = INTERPRETATION_NOTES['unvalidated_ai' if mode == 'ai' else 'rules']

    # -- image review ------------------------------------------------------

    def review_image(self, result: Dict, *, region: str, quality: Optional[Dict] = None, mode: str = 'ai') -> Dict:
        """Validate and finalize an image analysis before it is displayed or saved."""
        r = copy.deepcopy(result)
        for k in ('findings', 'not_assessable', 'possible_explanations', 'red_flags_present', 'red_flags', 'recommendations',
                  'avoid', 'see_doctor_when', 'safety_tips', 'observations', 'detected_conditions', 'information_gaps'):
            if not isinstance(r.get(k), list):
                r[k] = []
        if r.get('urgency') not in RANK:
            r['urgency'] = 'medium'
        if r.get('confidence') not in RANK:
            r['confidence'] = 'low'
        original = r['urgency']
        checks = ['schema_shape', 'image_quality_check', 'model_output_scan', 'language_check', 'medication_advice_check']
        adjustments: List[str] = []

        # separate observation from interpretation, and keep the legacy keys in sync
        if not r['findings'] and r['observations']:
            r['findings'] = [{'description': o, 'certainty': 'possibly visible'} for o in r['observations']]
        if not r['possible_explanations'] and r['detected_conditions']:
            r['possible_explanations'] = [{'name': c, 'likelihood': 'possible', 'why': '', 'supported_by': []} for c in r['detected_conditions'][:3]]
        for e in r['possible_explanations']:
            e.setdefault('why', '')
            e.setdefault('supported_by', [])

        lang = sanitize_language(r)

        q_rating = (quality or {}).get('rating')
        model_q = r.get('image_quality', 'fair')
        if q_rating:
            worse = reliability_worst(model_q, q_rating)
            if worse != model_q:
                adjustments.append(f'Photo quality was rated {worse} by our automatic check (the AI said {model_q}).')
            r['image_quality'] = worse
        quality_final = r.get('image_quality', 'fair')

        # a weak or non-medical photo can never support strong claims
        weak = quality_final == 'poor' or r.get('image_type') == 'not_medical'
        if weak and r['confidence'] != 'low':
            adjustments.append("The AI's confidence rating was lowered because the photo cannot support it.")
            r['confidence'] = 'low'
        if weak or r['confidence'] == 'low':
            for e in r['possible_explanations']:
                if e['likelihood'] == 'more likely':
                    e['likelihood'] = 'possible'
                    if 'likelihood_downgraded' not in checks:
                        checks.append('likelihood_downgraded')
            if 'likelihood_downgraded' in checks:
                adjustments.append('"More likely" labels were downgraded to "possible" because the evidence is weak.')

        floors = image_floors(r)
        d = self.decide(original, floors, False)
        final, care = d['level'], d['care_level']
        if d['raised']:
            adjustments.append(f"Urgency raised from {original} to {final} by safety rules: " + '; '.join(d['reasons'][:4]) + '.')
        r['urgency'] = final
        r['care_level'] = care
        r['timeframe'] = self.timeframe_for(care, r.get('timeframe', ''))
        r['urgency_reasons'] = d['reasons']

        if d['emergency']:
            labels = ['Severe bleeding'] if any('bleed' in x.lower() or 'spurt' in x.lower() or 'arterial' in x.lower() for x in d['reasons']) else []
            info = build_emergency(labels, region)
            r['emergency_alert'] = {'alert': True, 'message': EMERGENCY_MESSAGE, 'action': localize_text(EMERGENCY_ACTION, region),
                                    'reason': '; '.join(d['reasons'][:2])}
            r['recommendations'] = info['steps'] + [x for x in r['recommendations'] if x not in info['steps']]
            adjustments.append('Emergency guidance added because the analysis describes a dangerous finding.')
        else:
            r['emergency_alert'] = {'alert': False, 'message': '', 'action': '', 'reason': ''}

        if not r['not_assessable']:
            r['not_assessable'] = ['How deep it is, whether it is infected, and any pain, numbness or fever cannot be judged from a photo.']
        if not r['recommendations']:
            r['recommendations'] = ['Keep the area clean', 'Take another photo in better light', 'See a healthcare professional if you are worried']
        if not r['red_flags'] and r.get('warning_signs'):
            r['red_flags'] = list(r['warning_signs'])

        # legacy keys derived from the structured ones
        r['observations'] = [f['description'] for f in r['findings']]
        r['detected_conditions'] = [e['name'] for e in r['possible_explanations']]

        insufficient = r.get('image_type') == 'not_medical' or bool((quality or {}).get('unusable')) or r.get('assessment_status') == 'insufficient'
        r['assessment_status'] = 'insufficient' if insufficient else ('limited' if quality_final == 'poor' or mode == 'basic' else 'complete')
        rel = reliability.assess('image', mode=mode, ai_self_rating=r['confidence'] if mode == 'ai' else None, image_quality=quality_final,
                                 safety_overrode=d['raised'] and mode == 'ai', insufficient=insufficient,
                                 explanations=len(r['possible_explanations']))
        r['reliability'] = rel
        r['uncertainty_statement'] = reliability.uncertainty_statement(rel['level'])
        self._finish(r, mode)
        if quality:
            r['quality_check'] = quality

        if lang['softened']:
            adjustments.append('Overconfident wording was softened so the result is not presented as a diagnosis.')
        for kind in lang['removed']:
            adjustments.append(f'Removed {kind} from the advice (ask a pharmacist or doctor instead).')
        r['safety_review'] = {'engine_version': ENGINE_VERSION, 'checks': checks, 'adjustments': adjustments,
                              'original_urgency': original, 'final_urgency': final, 'urgency_raised': d['raised'],
                              'language_softened': lang['softened'], 'removed': lang['removed']}
        return r

    # -- safe fallbacks ----------------------------------------------------

    def insufficient_image(self, quality: Dict, region: str, reason: str = '') -> Dict:
        """Honest 'we can't assess this' result. No guessing, no reassurance."""
        n = localize_text(N, region)
        advice = list(quality.get('advice') or []) or ['Take a clear, close, well-lit photo of the affected area.']
        r = {
            'image_type': 'other_medical', 'image_quality': 'poor', 'quality_note': reason or ' '.join(quality.get('issues') or []),
            'summary': "This photo isn't clear enough to assess, so no conclusion is given. Nothing here means the problem is minor or serious.",
            'findings': [], 'not_assessable': ['Nothing reliable can be read from this photo.'], 'possible_explanations': [],
            'red_flags_present': [], 'red_flags': [], 'confidence': 'low', 'urgency': 'medium',
            'recommendations': advice + [f"Do not wait for a better photo if it is bleeding, very painful, spreading or you are worried: see a doctor, or call {n} in an emergency."],
            'avoid': [], 'see_doctor_when': ['It is painful, bleeding, swollen, spreading or getting worse', 'You are worried about it for any reason'],
            'safety_tips': [], 'information_gaps': [], 'assessment_status': 'insufficient',
            'analysis_mode': 'rules',
        }
        out = self.review_image(r, region=region, quality=quality, mode='basic')
        out['assessment_status'] = 'insufficient'
        return out

    # -- chat --------------------------------------------------------------

    def review_chat_reply(self, answer: str, region: str, analysis_level: Optional[str] = None, max_chars: int = 1800) -> Tuple[str, List[str]]:
        """Check a model chat reply before it is shown. Returns (answer, notes)."""
        notes: List[str] = []
        answer = localize_text(answer, region)
        had_reassurance = bool(_REASSURANCE_RE.search(answer))   # check before softening rewrites it
        answer, softened = soften(answer)
        if softened:
            notes.append('softened_language')
        if len(answer) > max_chars:
            cut = answer[:max_chars].rsplit('. ', 1)[0]
            answer = (cut + '.') if cut else answer[:max_chars]
            notes.append('truncated')
        if _DOSE_RE.search(answer):
            answer += ("\n\nI can't advise on medicine amounts or combinations. Please follow the label or ask a pharmacist or doctor, "
                       "especially for children, pregnancy or if you take other medicines.")
            notes.append('dose_note')
        if analysis_level in ('medium', 'high') and had_reassurance:
            answer += ("\n\nPlease still follow the advice in your earlier result" +
                       (": it suggested getting medical care today." if analysis_level == 'high' else ": it suggested seeing a doctor soon."))
            notes.append('reassurance_corrected')
        return answer, notes

    # -- final gate --------------------------------------------------------

    def final_gate(self, result: Dict, kind: str, region: str, *, text: str = '') -> Dict:
        """Defence in depth: make sure whatever is about to be displayed has been reviewed."""
        if not isinstance(result, dict) or result.get('error'):
            return result
        review = result.get('safety_review')
        if review and review.get('engine_version') == ENGINE_VERSION:
            return result
        if kind == 'image':
            return self.review_image(result, region=region, quality=result.get('quality_check'), mode='ai' if result.get('analysis_mode') == 'ai' else 'basic')
        return self.review_symptom(result, text=text, region=region, mode='ai' if result.get('analysis_mode') == 'ai' else 'basic')


def reliability_worst(a: str, b: str) -> str:
    order = {'good': 0, 'fair': 1, 'poor': 2}
    return a if order.get(a, 1) >= order.get(b, 1) else b


engine = TriageEngine()

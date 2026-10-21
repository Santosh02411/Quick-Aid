"""First-aid guides for the most common everyday problems, used by the built-in (no-AI) mode.

Why this exists: the keyword table in basic mode only knew *illness* words (fever, cough...). The most common reasons people open a first-aid app,
a cut, a burn, a sprain, a bite, got "Please describe your symptoms more specifically". Each guide below gives, in plain words: what it
probably is, what to do now (in order), what NOT to do, and exactly when to get it seen, plus rules that move the advice UP to "see a doctor today"
or "call the emergency number" when the description contains a danger sign.

These are standard, widely published first-aid steps (the kind in Red Cross / NHS / WHO guidance), written conservatively. They are general advice,
not a diagnosis, and the safety engine still reviews every result and can only make the advice MORE cautious.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

N = '{EMERGENCY_NUMBER}'
CARE_ORDER = {'self_care': 0, 'soon': 1, 'same_day': 2, 'emergency': 3}


def _rx(p: str) -> 're.Pattern':
    return re.compile(p, re.I)


@dataclass(frozen=True)
class Guide:
    id: str
    name: str                                  # shown as "what it could be"
    pattern: 're.Pattern'
    why: str                                   # one line: why this matches
    summary: str                               # one plain sentence for the top of the result
    steps: Tuple[str, ...]
    avoid: Tuple[str, ...]
    see_doctor: Tuple[str, ...]
    care: str = 'self_care'
    escalations: Tuple[Tuple['re.Pattern', str, str], ...] = ()      # (pattern, care level, reason shown to the person)

    def care_for(self, text: str) -> Tuple[str, List[str]]:
        """The care level for THIS description (the guide's own, raised by any danger sign in the text) and the reasons for raising it."""
        care, reasons = self.care, []
        for rx, level, reason in self.escalations:
            if rx.search(text) and CARE_ORDER[level] > CARE_ORDER[care]:
                care = level
            if rx.search(text) and CARE_ORDER[level] >= CARE_ORDER[self.care] and reason not in reasons:
                reasons.append(reason)
        return care, reasons


GUIDES: List[Guide] = [
    Guide('choking', 'Choking or something stuck in the throat',
          _rx(r"\b(chok(?:e|ed|es|ing)|something (?:is )?stuck in (?:my |his |her |their |the )?(?:throat|windpipe)|swallowed (?:a |an |some )?(?:coin|bone|toy|bead|battery|magnet|button))"),
          'You described choking or swallowing an object.', 'Choking can block breathing within minutes, so act now.',
          (f'Call {N} now, or ask someone else to, if the person cannot cough, speak or breathe.',
           'Adult or child over 1 year: give up to 5 firm back blows between the shoulder blades with the heel of your hand, leaning them forward.',
           'Then give up to 5 abdominal thrusts: stand behind them, fist just above the belly button, pull sharply inwards and upwards. Alternate 5 and 5 until the object comes out.',
           'Baby under 1 year: lay them face down along your forearm with the head lower than the body, give up to 5 firm back blows, then turn them over and give 5 chest thrusts with two fingers on the breastbone. Never do abdominal thrusts on a baby.',
           'If they become unresponsive and are not breathing normally, start CPR and keep going until help arrives.',
           'Anyone who swallowed a battery or magnet needs the emergency department even if they seem fine.'),
          ('Do not slap the back of a person who is coughing forcefully, let them keep coughing', 'Do not put your fingers in their mouth unless you can see the object', 'Do not give water or food'),
          ('Anyone who was choking and got the object out should still be checked by a doctor the same day if they have a cough, pain or trouble swallowing afterwards',),
          care='emergency'),
    Guide('broken_bone', 'Possible broken bone (fracture)',
          _rx(r"\b(?:broken|broke|fracture[ds]?|cracked)\b.{0,15}\b(?:bone|arm|leg|wrist|ankle|finger|toe|rib|collarbone|hip|foot|hand|nose|jaw|elbow|shoulder)s?\b|\b(?:arm|leg|wrist|ankle|finger|toe|rib|collarbone|hip|foot|hand|elbow|shoulder)\b.{0,15}\b(?:broken|fracture[ds]?|cracked)\b|\bbone (?:sticking|poking|showing)|\bsuspected fracture"),
          'You described an injury that may have broken a bone.', 'A suspected broken bone needs an X-ray. Keep it still until then.',
          ('Keep the injured part as still as you can. Support it in the position you found it, with a pillow, rolled clothes or a sling.',
           'Do not try to straighten it or push anything back in.',
           'If there is bleeding, press gently around (not on) the wound with a clean cloth. If a bone is visible, cover loosely with a clean cloth.',
           'Put a cold pack wrapped in cloth on the area for 15 to 20 minutes to ease pain and swelling.',
           'Take off rings, watches and tight things from the injured limb before it swells.',
           'Do not eat or drink in case you need an operation.',
           f'Go to a hospital or call {N}. Do not walk on a suspected broken leg or ankle.'),
          ('Do not move someone with a suspected neck, back or hip injury', 'Do not massage it or apply heat', 'Do not wait for it to get better by itself'),
          ('Always, for a suspected fracture: the same day at the latest', 'Straight away if the hand or foot below the injury is numb, pale, blue or cold'),
          care='same_day',
          escalations=((_rx(r"bone (?:sticking|poking|showing|visible)|open fracture|spurting|\b(?:neck|spine|back|hip|skull)\b|numb|pale|blue|cold fingers|cold toes|can'?t feel"), 'emergency', 'A sign of a serious fracture or nerve and blood-vessel injury'),)),
    Guide('animal_bite', 'Animal bite or scratch',
          _rx(r"\b(?:dog|cat|monkey|rat|bat|puppy|kitten|stray|animal|squirrel|rabbit|horse|cow|buffalo|goat|pig|donkey|camel|mongoose|jackal|fox)\b.{0,25}\b(?:bit|bite[ds]?|bitten|scratch(?:ed)?|licked)\b|\b(?:bit|bitten|bite) by (?:a |an |the )?(?:dog|cat|monkey|rat|bat|animal|stray)\b|\bdog bite|\bcat scratch|\banimal bite|\bmonkey bite"),
          'You described being bitten or scratched by an animal.', 'Animal bites can carry rabies and tetanus, so washing now and seeing a doctor today matter.',
          ('Wash the wound with soap and running water for at least 15 minutes. This is the single most important step.',
           'Press a clean cloth on it if it is bleeding. Then apply an antiseptic such as povidone-iodine if you have one.',
           'Cover loosely with a clean dressing. Do not stitch or seal it tightly.',
           'See a doctor today, even if the wound looks small. You may need a rabies vaccine (and rabies immunoglobulin for deep or high-risk bites) and a tetanus booster. Rabies treatment works only if started early.',
           'If you can, note the animal and whether it can be watched for 10 days, but never try to catch it.'),
          ('Do not apply chilli, turmeric, oil, herbs or any home remedy to the wound', 'Do not close or bandage it tightly', 'Do not wait to see if the animal falls sick'),
          ('Today, for every animal bite or a scratch that broke the skin', 'Straight away if the bite is on the face, neck, hand or joint, is deep, bleeds heavily, or the animal was a bat, monkey or stray'),
          care='same_day'),
    Guide('burn', 'Burn or scald',
          _rx(r"\b(?:burn(?:s|ed|t)?(?!\s+out)|scald(?:s|ed)?|boiling (?:water|oil|milk)|hot (?:water|oil|tea|milk|steam|iron|pan|stove)|touched (?:the )?(?:hot|stove|iron|flame))\b"),
          'You described a burn or scald.', 'Cool it with running water straight away. Every second counts.',
          ('Cool the burn under cool (not ice-cold) running water for 20 minutes. Start as soon as possible.',
           'Take off rings, bracelets, watches and tight clothing near the burn before it swells, unless they are stuck to the skin.',
           'Cover it loosely with clean cling film or a clean, non-fluffy cloth. Lay the film on, do not wrap it tightly.',
           'Keep the person warm, and keep the burned area raised if you can.',
           'Pain relief such as paracetamol is fine if you normally take it. Follow the label.',
           'Keep it clean and dry afterwards. A small burn usually heals in 1 to 2 weeks.'),
          ('Do not use ice, iced water, butter, oil, toothpaste or any cream on a fresh burn', 'Do not burst blisters', 'Do not stick plasters or cotton wool on it', 'Do not remove clothing that is stuck to the burn'),
          ('The burn is bigger than the palm of the person’s hand', 'It is on the face, hands, feet, groin, a joint or goes all the way round a limb', 'It looks white, leathery, charred, or the person feels no pain there',
           'It is a chemical or electrical burn, or the person breathed in smoke or steam', 'Blisters are large or it shows signs of infection (spreading redness, pus, fever)', 'The person is a baby, young child or older adult'),
          care='self_care',
          escalations=((_rx(r"\b(?:face|eye|hand|hands|finger|fingers|foot|feet|toe|toes|genital|groin|joint|knuckle|palm|neck|blister|blisters|child|baby|infant|toddler|big|large|oil)\b"), 'soon', 'Burns on the hands, face, feet or joints, with blisters, or in children need a doctor'),
                       (_rx(r"\b(?:white|charred|black|leathery|deep|electric(?:al)?|chemical|acid|bleach|whole|entire|covering|huge|very large|inhaled|smoke)\b"), 'same_day', 'A deep, large, chemical or electrical burn needs urgent care'))),
    Guide('cut', 'Minor cut or scrape',
          _rx(r"\b(?:cut(?!\s+(?:down|back|out|off|short|in line))|cuts|scrape[ds]?|graze[ds]?|scratch(?:ed|es)|abrasion|nick(?:ed)?|gash(?:ed)?|small wound|paper cut|minor wound|knife (?:slip|cut)|sliced)\b"),
          'You described a cut, scrape or small wound.', 'Stop the bleeding, clean it well, and cover it.',
          ('Wash your hands, or put on clean gloves if you have them.',
           'Press a clean cloth firmly on the cut for 5 to 10 minutes to stop the bleeding. Raise the injured part above the heart if you can. Do not keep lifting the cloth to look.',
           'Rinse the cut under clean running water for a few minutes. Wash around it gently with mild soap, keeping soap out of the wound. Remove loose dirt.',
           'Pat dry with a clean cloth. A thin layer of antiseptic ointment is optional.',
           'Cover with a clean plaster or dressing. Change it daily, and any time it gets wet or dirty.',
           'Check that your tetanus vaccination is up to date (booster every 10 years, sooner for a dirty wound).'),
          ('Do not pour hydrogen peroxide, alcohol or iodine straight into the wound, they slow healing', 'Do not blow on it or touch it with unwashed hands', 'Do not pull out anything that is deeply stuck in the wound'),
          ('Bleeding does not stop after 10 minutes of firm, steady pressure', 'The cut is deep, gaping, long, or you can see fat, muscle or bone', 'Dirt, glass or a rusty object is in it and you cannot get it out',
           'It is on the face, over a joint, or you cannot move or feel a finger or toe', 'It came from a bite or a dirty or rusty object', 'Signs of infection: spreading redness, warmth, swelling, pus, red streaks, or fever'),
          care='self_care',
          escalations=((_rx(r"\b(?:deep|gaping|gaped|won'?t stop|not stopping|can'?t stop|keeps? bleeding|spurting|pumping|bone|tendon|muscle|numb|can'?t (?:move|feel)|rusty|dirty|glass|embedded|stuck in|stitches?|face|eye|lip|joint)\b"), 'same_day', 'A deep, dirty, bleeding-heavily or face/joint wound needs to be seen today'),
                       (_rx(r"\b(?:infected|pus|red streaks?|swollen|swelling|hot to touch|fever)\b"), 'soon', 'Signs of infection need a doctor'))),
    Guide('nosebleed', 'Nosebleed',
          _rx(r"\b(?:nose ?bleeds?|bleeding (?:from )?(?:my |the |his |her )?nose|nose is bleeding|bloody nose)\b"),
          'You described a nosebleed.', 'Most nosebleeds stop with pressure in 10 to 15 minutes.',
          ('Sit upright and lean slightly forward, so blood does not run down your throat.',
           'Pinch the soft part of the nose, just below the bony bridge, firmly with thumb and finger. Breathe through your mouth.',
           'Keep pinching for a full 10 to 15 minutes without letting go to check. Use a timer.',
           'You can put a cold pack on the bridge of the nose or the back of the neck.',
           'Spit out any blood that reaches your mouth. Stay calm and still.',
           'Afterwards, do not blow, pick or rub your nose for 24 hours, and avoid hot drinks and heavy lifting for a day.'),
          ('Do not tilt your head back or lie flat', 'Do not stuff tissue or cotton deep in the nose', 'Do not blow your nose afterwards'),
          ('It has not stopped after 20 minutes of pinching, or the bleeding is heavy', 'It follows a blow to the head or face', 'You take blood thinners, or have a bleeding disorder, or nosebleeds keep coming back', 'You feel faint or dizzy'),
          care='self_care',
          escalations=((_rx(r"\b(?:won'?t stop|not stopping|can'?t stop|heavy|gushing|20 minutes|half an hour|after (?:a |an )?(?:fall|blow|hit|punch|accident|injury)|hit (?:my )?(?:head|face)|blood thinner|warfarin|aspirin|faint|dizzy)\b"), 'same_day', 'A nosebleed that will not stop, follows an injury, or comes with blood thinners needs to be seen today'),)),
    Guide('head_bump', 'Bump or blow to the head',
          _rx(r"\b(?:hit|bump(?:ed)?|knock(?:ed)?|bang(?:ed)?|struck|fell on|smashed)\b.{0,12}\b(?:my |his |her |their |the )?(?:head|forehead|skull)\b|\bhead (?:injury|bump|knock|hit|blow)|\bbump on (?:my |the |his |her )?head"),
          'You described a blow to the head.', 'Most head bumps are minor, but they need to be watched closely for 24 hours.',
          ('Hold a cold pack wrapped in cloth on the bump for 10 to 20 minutes to reduce swelling.',
           'Rest quietly. Stay with someone who can watch for changes for the next 24 hours.',
           'Do not drink alcohol, and avoid sports or hard activity until fully recovered.',
           'Paracetamol is fine for headache if you normally take it. Avoid aspirin and ibuprofen for the first day.',
           'If it is a child, keep them calm and awake and watch how they behave compared with normal.',
           f'Call {N} or go to the emergency department at once if any of these appear: vomiting more than once, drowsiness or hard to wake, confusion, severe or worsening headache, fainting, a fit, weakness or numbness, blurred or double vision, clear fluid or blood from the nose or ears, or unequal pupils.'),
          ('Do not let them sleep unwatched in the first hours without checking they wake normally', 'Do not drive or climb', 'Do not give alcohol or sedatives'),
          ('Any of the warning signs listed in the steps', 'A baby or toddler hit their head, or an older person on blood thinners did', 'The fall was from a height, in a road accident, or the person lost consciousness even briefly'),
          care='soon',
          escalations=((_rx(r"\b(?:vomit(?:ed|ing)?|sleepy|drows\w*|confus\w*|faint(?:ed)?|pass(?:ed)? out|unconscious|seizure|fit|fitting|blurr?y|double vision|clear fluid|bleeding from (?:the )?(?:ear|nose)|baby|infant|toddler|can'?t remember|memory|slurr\w*|weak(?:ness)?|numb|worsening headache|severe headache|blood thinner|warfarin)\b"), 'same_day', 'After a head injury these signs mean you should be checked straight away'),)),
    Guide('sprain', 'Sprain or strain (twisted joint or pulled muscle)',
          _rx(r"\b(?:sprain(?:ed|s)?|twist(?:ed)?|rolled|pulled|strain(?:ed)?)\b.{0,25}\b(?:ankle|wrist|knee|muscle|back|foot|thumb|finger|hamstring|calf|groin|shoulder|neck|elbow)\b|\b(?:ankle|wrist|knee|muscle|back|foot|thumb|hamstring|calf|elbow)\b.{0,15}\b(?:sprain|twist|rolled|pulled|strain)|\bswollen (?:ankle|wrist|knee)"),
          'You described a twisted joint or pulled muscle.', 'Rest, cold, support and raising it usually work for a mild sprain. Do not put weight on it if it hurts a lot.',
          ('Rest the joint. Stop what you were doing and avoid putting weight on it for the first day or two.',
           'Apply a cold pack wrapped in a cloth for 15 to 20 minutes, every 2 to 3 hours, for the first 2 days.',
           'Wrap it with a snug elastic bandage, firm but not tight. If toes or fingers tingle, go numb or turn blue, loosen it.',
           'Raise the injured part above heart level when sitting or lying down.',
           'Paracetamol helps with pain. Follow the label.',
           'After 2 days, start gentle movement within comfort. Return to sport slowly once it moves normally without pain.'),
          ('Do not use heat, hot baths or massage in the first 2 days, they increase swelling', 'Do not drink alcohol', 'Do not walk or run on it if it hurts a lot'),
          ('You cannot put weight on it or take 4 steps', 'It looks bent or out of shape, or you heard a snap or pop', 'It is numb, tingling, pale, cold or blue', 'Severe pain, or no improvement after 3 to 5 days'),
          care='self_care',
          escalations=((_rx(r"\b(?:(?:can'?t|cannot|unable to) (?:put weight|walk|bear|move|stand)|deform\w*|bent|out of shape|numb|pins and needles|cold|blue|pale|heard a (?:pop|snap|crack)|popped|snapped|bone)\b"), 'same_day', 'You may have a fracture or serious ligament injury, get an X-ray today'),
                       (_rx(r"\b(?:severe|terrible|unbearable|very swollen|badly swollen|not getting better|days)\b"), 'soon', 'Severe or lasting pain needs a check'))),
    Guide('eye', 'Something in the eye, or an eye injury',
          _rx(r"\b(?:something|dust|sand|dirt|grit|eyelash|particle|insect|fly|hair|speck|foreign (?:object|body))\b.{0,15}\b(?:in|into)\b.{0,10}\b(?:my |the |his |her )?eyes?\b|\beyes?\b.{0,20}\b(?:chemical|bleach|acid|detergent|cleaner|splash(?:ed)?|pepper spray|soap|shampoo|lime|paint)\b|\b(?:chemical|bleach|acid|detergent|splash(?:ed)?|shampoo|soap|lime)\b.{0,20}\b(?:in|into|got in)\b.{0,10}\beyes?\b|\bscratched (?:my )?(?:eye|cornea)|\bpoked (?:in )?(?:my |the )?eye"),
          'You described something in your eye or an eye injury.', 'Do not rub it. Rinse with clean water first.',
          ('Do not rub the eye.',
           'Wash your hands, then rinse the eye with clean running water or clean saline for at least 15 minutes, tilting the head so the water runs away from the other eye.',
           'For a small speck: blink several times, or pull the upper lid over the lower lid so the lashes brush it out.',
           'For chemicals (bleach, cleaner, acid, lime, pepper spray): keep rinsing for 20 minutes without stopping, take out contact lenses, and go to a hospital straight away.',
           'Cover the eye loosely with a clean pad if it hurts, and see a doctor if anything stays or the eye was cut or poked.'),
          ('Do not rub it', 'Do not use cotton buds, tweezers or anything pointed in the eye', 'Do not try to remove something stuck on the eyeball or embedded in it', 'Do not use eye drops or ointments unless a doctor says so'),
          ('A chemical got in the eye: after rinsing, go to the hospital straight away', 'The eye still hurts, is red, waters a lot or vision is blurred after rinsing', 'Something is stuck on or in the eye, or it was poked or cut', 'You wear contact lenses and the eye is painful or red'),
          care='soon',
          escalations=((_rx(r"\b(?:chemical|bleach|acid|detergent|cleaner|pepper spray|lime|paint|poked|stabbed|embedded|stuck in|cut|blurr?y|vision|sharp|metal|glass)\b"), 'same_day', 'A chemical splash, a poked or cut eye, or lost vision needs urgent eye care today'),)),
    Guide('insect', 'Insect bite or sting',
          _rx(r"\b(?:mosquito|bee|wasp|hornet|ant|bug|insect|spider|tick|midge|flea|bed ?bug|sandfly|gnat|horsefly|caterpillar)s?\b.{0,20}\b(?:bit(?:e|ten|es)?|sting|stung|stings)\b|\b(?:bite[sd]?|sting|stung|stings)\b.{0,12}\b(?:by|from) (?:a |an )?(?:mosquito|bee|wasp|hornet|ant|insect|spider|tick|flea|bug|caterpillar)|\b(?:bee|wasp|hornet) sting|\binsect (?:bite|sting)|\bmosquito bites?\b|\bbed ?bug bites?\b|\b(?:bitten|bit|stung)\b.{0,12}\bby (?:a |an |the )?(?:mosquito|bee|wasp|hornet|ant|insect|spider|tick|flea|bug|caterpillar|scorpion fly)\b"),
          'You described an insect bite or sting.', 'Most bites and stings are just itchy and sore and settle in a few days.',
          ('If a sting is still in the skin (bee), scrape it out sideways with a card or fingernail. Do not squeeze it.',
           'Wash the area with soap and water.',
           'Put a cold pack or a cool damp cloth on it for 10 to 15 minutes to reduce swelling and itch.',
           'An antihistamine cream or tablet, or a mild hydrocortisone cream, can ease itching. Follow the label.',
           'Try not to scratch: it can cause infection. Keep nails short and the area clean.',
           'If it is a tick: grip it with fine tweezers as close to the skin as possible and pull straight up, steadily. Clean the skin afterwards. Do not squash, burn or smother it.'),
          ('Do not scratch', 'Do not squeeze a bee sting', 'Do not put raw onion, toothpaste, oil or other home remedies on the bite'),
          ('Swelling spreads a lot, or the area becomes very red, hot, painful, or has pus or red streaks', 'You have fever, a rash all over, or you feel unwell after a tick bite',
           'You were stung many times, or in the mouth or throat', 'You have ever had a serious allergic reaction to stings'),
          care='self_care',
          escalations=((_rx(r"\b(?:swell(?:ing|ed|s)?\b.{0,20}\b(?:face|lips?|tongue|throat|eye)|(?:face|lips?|tongue|throat|eyelid)\b.{0,15}\bswell\w*|many|dozens|hundreds|swarm|in (?:my |the )?mouth|throat|tongue|spreading|pus|red streaks?|allerg\w*|hives|wheez\w*|dizzy|faint)\b"), 'same_day', 'Swelling of the face, lips or throat, many stings, or spreading redness need urgent care'),
                       (_rx(r"\b(?:tick|fever|rash|unwell|swollen|infected)\b"), 'soon', 'A tick bite, fever or an infected bite needs a doctor'))),
    Guide('sunburn', 'Sunburn',
          _rx(r"\b(?:sun ?burn(?:t|ed|s)?|sunburnt|too much sun|burn(?:t|ed)? (?:from|by) the sun|sun poisoning)\b"),
          'You described sunburn.', 'Cool the skin, moisturise and drink plenty of water. Mild sunburn heals in about a week.',
          ('Get out of the sun. Cool the skin with a cool (not icy) shower, bath or damp cloth for 10 to 15 minutes.',
           'Apply a gentle moisturiser or aloe vera gel several times a day.',
           'Drink extra water to replace fluid lost through the skin.',
           'Paracetamol or ibuprofen (if you can take it) helps pain. Follow the label.',
           'Cover the area with loose clothing and keep it out of the sun until healed.'),
          ('Do not burst blisters', 'Do not use butter, oil or petroleum jelly on fresh sunburn', 'Do not go back into strong sun until it has healed'),
          ('Blisters cover a large area, or the burn is severe and painful', 'Fever, chills, headache, dizziness, confusion or feeling faint', 'The sunburn is on a baby or young child'),
          care='self_care',
          escalations=((_rx(r"\b(?:blister(?:s|ed|ing)?|fever|chills|dizz\w+|faint\w*|confus\w*|vomit\w*|baby|infant|toddler|child|severe|swollen face)\b"), 'soon', 'Blistering, fever or dizziness after sunburn, or sunburn in a child, needs a doctor'),)),
    Guide('heat', 'Heat exhaustion or heatstroke',
          _rx(r"\b(?:heat ?stroke|heat exhaustion|sun ?stroke|overheat(?:ed|ing)?|heat cramps?|collapsed in the heat|dizzy in the heat|too hot)\b|\bheat (?:exhaustion|stroke)|\bfeel(?:ing)? faint in the (?:heat|sun)"),
          'You described feeling unwell from heat.', 'Cool down quickly. Heatstroke is an emergency.',
          ('Move to a cool, shaded place or an air-conditioned room. Lie down and raise the legs a little.',
           'Take off extra clothing. Cool the skin with cool water, wet cloths or a fan, especially the neck, armpits and groin.',
           'If they are awake and not vomiting, give small sips of cool water or oral rehydration solution.',
           'Stay with them. Rest until they feel completely better, and avoid heat and exertion for the rest of the day.',
           f'If they are confused, faint, stop sweating, have hot dry skin, a fit, or do not improve in 15 minutes of cooling, call {N}: this is heatstroke and is life-threatening.'),
          ('Do not give anything to drink if they are confused or drowsy', 'Do not give alcohol, tea or coffee', 'Do not wait: cool them straight away'),
          ('They are confused, faint or have a fit', 'No improvement after 30 minutes of cooling', 'The person is a baby, young child or older adult'),
          care='soon',
          escalations=((_rx(r"\b(?:confus\w*|faint(?:ed|ing)?|unconscious|collapsed|not sweating|stopped sweating|hot (?:and )?dry|seizure|fit|fitting|delirious|slurred|40 ?°?c|104)\b"), 'emergency', 'Confusion, fainting, a fit or not sweating in the heat is heatstroke, an emergency'),)),
    Guide('splinter', 'Splinter or thorn',
          _rx(r"\b(?:splinter|thorn|sliver|prickle|cactus|fish ?bone|glass shard)s?\b"),
          'You described a splinter or thorn.', 'Most can be removed safely at home with clean tweezers.',
          ('Wash your hands and the area with soap and water.',
           'Clean a pair of tweezers with alcohol. If the end of the splinter sticks out, grip it and pull slowly out along the angle it went in.',
           'If it is under the skin, a needle cleaned with alcohol can lift the skin over the tip. Then pull it out with the tweezers.',
           'Squeeze the spot gently so it bleeds a little, which helps flush out germs. Wash again and cover with a plaster.',
           'Check that your tetanus vaccination is up to date.'),
          ('Do not dig deep or squeeze hard', 'Do not try to remove splinters near or in the eye', 'Do not leave a deep or dirty splinter in'),
          ('You cannot remove all of it, or it is deep, or in the eye, nail bed or a joint', 'The area becomes red, swollen, warm, painful or has pus', 'You have diabetes or a weak immune system'),
          care='self_care',
          escalations=((_rx(r"\b(?:deep|eye|nail|joint|infected|pus|swollen|red streaks?|diabet\w*|can'?t (?:get|remove)|glass)\b"), 'soon', 'A deep, infected or hard-to-remove splinter needs a doctor'),)),
    Guide('blister', 'Blister',
          _rx(r"\b(?:blister(?:s|ed)?|blistering|friction (?:sore|burn)|rubbed (?:a )?(?:sore|raw))\b"),
          'You described a blister.', 'Leave it alone if you can. The skin on top protects the healing skin.',
          ('Do not burst it. Keep it clean and covered with a soft plaster or a ring of padding around it.',
           'Avoid whatever caused it: change shoes, wear thick socks, or rest the area.',
           'If it has burst, wash gently with soap and water, leave the skin flap in place, apply antiseptic ointment and cover with a clean dressing.',
           'Change the dressing daily and watch for infection.'),
          ('Do not pop or peel it', 'Do not use alcohol or iodine on a broken blister', 'Do not keep walking on a painful blister in tight shoes'),
          ('It is large, very painful, or on the palm or sole', 'Signs of infection: pus, spreading redness, warmth, red streaks, fever', 'You have diabetes or poor circulation', 'It came from a burn, chemical, or you do not know why you have blisters'),
          care='self_care',
          escalations=((_rx(r"\b(?:infected|pus|red streaks?|fever|diabet\w*|chemical|all over|many|no reason|unknown|spreading)\b"), 'soon', 'Infected blisters, blisters in diabetes, or blisters with no clear cause need a doctor'),)),
    Guide('bruise', 'Bruise',
          _rx(r"\b(?:bruise[ds]?|bruising|contusion|black and blue|bumped (?:my|the) (?:leg|arm|knee|elbow|shin|hip|shoulder|foot|hand)|knock(?:ed)? (?:my|the) (?:leg|arm|knee|elbow|shin|hip|shoulder|foot|hand))\b"),
          'You described a bruise or a knock.', 'Most bruises fade in 2 weeks. Cold early on limits swelling.',
          ('Put a cold pack wrapped in a cloth on it for 10 to 20 minutes, several times in the first day.',
           'Rest the area and raise it if you can.',
           'After 2 days, a warm compress can help the bruise fade.',
           'Paracetamol helps with pain. Follow the label.'),
          ('Do not rub or massage it hard', 'Do not apply heat in the first 2 days'),
          ('The bruise is very large, very painful or swollen, or you cannot move the limb', 'You get bruises without injury, or very easily, or you take blood thinners', 'It has not faded after 2 weeks', 'The bruise follows a blow to the head, chest or belly'),
          care='self_care',
          escalations=((_rx(r"\b(?:huge|very large|very painful|very swollen|can'?t move|no reason|for no reason|without (?:any )?(?:injury|reason|cause)|keeps? (?:appearing|coming)|easily|blood thinner|warfarin|not fad\w*|belly|abdomen|chest|ribs?)\b"), 'soon', 'A large, painful or unexplained bruise, or bruising on blood thinners, needs a doctor'),)),
]
BY_ID = {g.id: g for g in GUIDES}


def match(text: str, limit: int = 2) -> List[Guide]:
    """The guides that fit this description, most specific first (the earliest mention in the text wins; at most `limit`)."""
    found = []
    for g in GUIDES:
        m = g.pattern.search(text or '')
        if m:
            found.append((m.start(), GUIDES.index(g), g))
    found.sort(key=lambda x: (CARE_ORDER[x[2].care_for(text)[0]] * -1, x[0], x[1]))          # the most urgent guide first
    return [g for _, _, g in found[:limit]]


_LEVEL_OF = {'self_care': 'low', 'soon': 'medium', 'same_day': 'high', 'emergency': 'high'}
_LEVEL_RANK = {'low': 0, 'medium': 1, 'high': 2}


def apply(result: dict, guides: List[Guide], text: str, localize) -> dict:
    """Merge first-aid guides into a result dict (in place; also returned). `localize` fills in the emergency number.

    * what to DO comes first (the guide's steps), ahead of any keyword-based illness advice;
    * 'avoid' and 'see a doctor when' come from the guide; the care level is raised (never lowered) by the guide and its danger-sign rules;
    * an emergency guide (choking, heatstroke with confusion, an open fracture) sets the emergency alert;
    * choking REPLACES the generic "breathing difficulty" steps, because sitting upright and using an inhaler would be wrong for a choking person.
    """
    if not guides:
        return result
    from quickaid import i18n
    steps, avoid, see, names, reasons, rank = [], [], [], [], [], 0
    for g in guides:
        care, why = g.care_for(text)
        rank = max(rank, CARE_ORDER[care])
        reasons += [x for x in why if x not in reasons]
        steps += [localize(x) for x in g.steps if localize(x) not in steps]
        avoid += [x for x in g.avoid if x not in avoid]
        see += [x for x in g.see_doctor if x not in see]
        names.append(g)
    care_name = [k for k, v in CARE_ORDER.items() if v == rank][0]
    explanations = [{'name': g.name, 'likelihood': 'more likely', 'why': g.why, 'supported_by': []} for g in names]
    old = [e for e in result.get('possible_explanations') or [] if e.get('name') not in {g.name for g in names}]
    result['possible_explanations'] = (explanations + old)[:5]
    result['possible_conditions'] = [e['name'] for e in result['possible_explanations']]
    replace_generic = any(g.id == 'choking' for g in guides)
    base = list(result.get('recommendations') or [])
    if replace_generic:
        base = [] if i18n.current() == 'en' else base[:1]            # in other languages keep the translated "call now" line
    result['recommendations'] = steps + [r for r in base if r not in steps]
    result['avoid'] = avoid + [x for x in result.get('avoid') or [] if x not in avoid]
    result['see_doctor_when'] = see + [x for x in result.get('see_doctor_when') or [] if x not in see]
    result['red_flags'] = (see[:4] + list(result.get('red_flags') or []))[:6]
    result['urgency_reasons'] = list(dict.fromkeys(list(result.get('urgency_reasons') or []) + reasons))
    level = _LEVEL_OF[care_name]
    if _LEVEL_RANK[level] > _LEVEL_RANK.get(result.get('urgency_level', 'low'), 0):
        result['urgency_level'] = level
    if care_name == 'emergency':
        result['emergency_alert'] = {'alert': True, 'reason': '; '.join([names[0].name] + reasons[:1]), 'message': '', 'action': ''}
    shown = [g.id.replace('_', ' ') for g in names]
    if not result.get('detected_symptoms'):
        result['detected_symptoms'] = shown
        result['reported'] = {'symptoms': shown, 'details': []}
    result['summary'] = (f"Most likely: {' / '.join(g.name for g in names[:2])}. The steps below are standard first aid, in order, followed by when to get it checked.")
    result['_first_aid_count'] = len(steps)                         # read (and removed) by the safety engine, which decides the final order
    result['_replace_generic_emergency_steps'] = replace_generic
    return result

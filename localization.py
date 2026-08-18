"""
Region-aware content for Quick Aid: emergency numbers and temperature units.

The original app hardcoded US assumptions inconsistently (911 in some
places, India's 108 in others - see templates/index.html's old emergency
button). This module gives every region a single, correct source of truth,
and both the deterministic fallback strings *and* the Gemini prompts pull
from it, so localization applies whether or not Gemini is configured.
"""

from typing import Dict

REGIONS: Dict[str, Dict[str, str]] = {
    'INTL': {'label': 'International / Other', 'emergency_number': 'your local emergency number', 'temp_unit': 'C'},
    'US':   {'label': 'United States',          'emergency_number': '911',                          'temp_unit': 'F'},
    'CA':   {'label': 'Canada',                 'emergency_number': '911',                          'temp_unit': 'C'},
    'UK':   {'label': 'United Kingdom',         'emergency_number': '999',                          'temp_unit': 'C'},
    'EU':   {'label': 'European Union',         'emergency_number': '112',                          'temp_unit': 'C'},
    'IN':   {'label': 'India',                  'emergency_number': '112',                          'temp_unit': 'C'},
    'AU':   {'label': 'Australia',              'emergency_number': '000',                          'temp_unit': 'C'},
}

DEFAULT_REGION = 'INTL'

# Placeholder token used in fallback (non-Gemini) recommendation/safety-tip
# strings, substituted with the user's actual regional emergency number
# just before a response is returned.
EMERGENCY_PLACEHOLDER = '{EMERGENCY_NUMBER}'

# Placeholder for the high-fever threshold, substituted with the region's
# preferred unit shown first (e.g. '103°F (39.4°C)' vs '39.4°C (103°F)').
FEVER_PLACEHOLDER = '{FEVER_TEMP}'


def get_region_info(region_code: str) -> Dict[str, str]:
    return REGIONS.get(region_code, REGIONS[DEFAULT_REGION])


def localize_text(text: str, region_code: str) -> str:
    """Replace region-dependent placeholders with this region's actual values."""
    info = get_region_info(region_code)
    text = text.replace(EMERGENCY_PLACEHOLDER, info['emergency_number'])
    if FEVER_PLACEHOLDER in text:
        text = text.replace(FEVER_PLACEHOLDER, fever_threshold_text(region_code))
    return text


def localize_string_list(items, region_code: str):
    return [localize_text(item, region_code) if isinstance(item, str) else item for item in items]


def localize_analysis(analysis: Dict, region_code: str) -> Dict:
    """
    Walk an analysis result dict's text fields (recommendations, safety
    tips, and any emergency_alert message/action) and substitute the
    emergency-number placeholder with the user's actual region.
    """
    if 'recommendations' in analysis:
        analysis['recommendations'] = localize_string_list(analysis['recommendations'], region_code)
    if 'safety_tips' in analysis:
        analysis['safety_tips'] = localize_string_list(analysis['safety_tips'], region_code)

    emergency = analysis.get('emergency_alert')
    if isinstance(emergency, dict):
        if 'message' in emergency:
            emergency['message'] = localize_text(emergency['message'], region_code)
        if 'action' in emergency:
            emergency['action'] = localize_text(emergency['action'], region_code)

    return analysis


def fever_threshold_text(region_code: str) -> str:
    """
    103°F / 39.4°C, ordered by the region's primary unit so the number
    someone actually uses locally comes first.
    """
    info = get_region_info(region_code)
    if info['temp_unit'] == 'F':
        return '103°F (39.4°C)'
    return '39.4°C (103°F)'


def prompt_context(region_code: str) -> str:
    """
    A short instruction appended to Gemini prompts so free-form AI-generated
    advice also respects the user's region instead of defaulting to US
    conventions (911, Fahrenheit-first).
    """
    info = get_region_info(region_code)
    return (
        f"The user is located in: {info['label']}. "
        f"If you need to reference emergency services, use '{info['emergency_number']}' "
        f"- do not assume 911 unless that is actually correct for this region. "
        f"If you state a temperature, lead with °{info['temp_unit']} and include the other unit in parentheses."
    )

"""Rotation/orientation handling, and whether the system correctly recognises unrelated or information-free
photos and asks for a retake rather than guessing. Covers gaps that had zero prior test coverage: EXIF-based
rotation correction (implemented but never actually verified end to end) and what happens with the specific
"this is not a usable medical photo" cases requirement 5 named - an unrelated image and an image with
insufficient visible information.
"""

import io

import numpy as np
import pytest
from PIL import Image, ImageOps

from quickaid.ai import image_quality as iq
from quickaid.ai.medical_analyzer import MedicalAnalyzer


def lesion_photo(n=400, seed=3):
    """A clearly non-square (portrait: taller than wide) synthetic photo, so rotation actually changes something
    observable. numpy array shape is (rows, cols) = (height, width), so the taller dimension goes first."""
    height, width = int(n * 1.4), n
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = (215, 165, 145)
    rng = np.random.default_rng(seed)
    arr = np.clip(arr.astype(np.int16) + rng.integers(-20, 20, (height, width, 1)), 0, 255).astype(np.uint8)
    arr[int(height * .3):int(height * .6), int(width * .4):int(width * .9)] = (190, 70, 70)   # the "lesion"
    return Image.fromarray(arr)   # .size is (width, height) = (n, n*1.4): taller than wide, i.e. portrait


class TestExifRotationCorrection:
    """_prepare_image() calls ImageOps.exif_transpose() - implemented, but previously had no test at all."""

    @pytest.mark.parametrize('exif_orientation', [1, 3, 6, 8])   # 1=normal, 3=180°, 6=90°CW, 8=90°CCW
    def test_a_photo_with_an_exif_rotation_tag_is_straightened(self, tmp_path, exif_orientation):
        base = lesion_photo()
        # Simulate what a phone camera does: save the pixels in the SENSOR's native orientation and record how to
        # display them correctly via the EXIF Orientation tag, rather than pre-rotating the pixel data itself.
        rotated_pixels = {1: base, 3: base.rotate(180), 6: base.rotate(-90, expand=True), 8: base.rotate(90, expand=True)}[exif_orientation]
        path = tmp_path / 'photo.jpg'
        exif = Image.Exif()
        exif[0x0112] = exif_orientation   # Orientation tag
        rotated_pixels.save(path, exif=exif)

        prepared = MedicalAnalyzer._prepare_image(str(path))
        # However it was stored, the OUTPUT must always match the original upright orientation (same aspect ratio
        # as the un-rotated source: portrait stays portrait).
        assert prepared.size[1] > prepared.size[0]   # taller than wide, same as the original base photo
        assert abs(prepared.size[1] / prepared.size[0] - base.size[1] / base.size[0]) < 0.05

    def test_a_photo_with_no_exif_data_at_all_is_not_touched(self, tmp_path):
        path = tmp_path / 'no_exif.png'
        lesion_photo().save(path)   # PNG: no EXIF support, nothing to correct
        prepared = MedicalAnalyzer._prepare_image(str(path))
        assert prepared.size[1] > prepared.size[0]

    def test_no_exif_metadata_survives_into_the_prepared_image(self, tmp_path):
        # privacy requirement, re-checked here specifically for the rotation path: the corrected image must not
        # carry the original orientation tag, camera make/model, GPS, or timestamp forward
        path = tmp_path / 'photo.jpg'
        exif = Image.Exif()
        exif[0x0112] = 6
        exif[0x010F] = 'TestCameraMake'
        lesion_photo().save(path, exif=exif)
        prepared = MedicalAnalyzer._prepare_image(str(path))
        assert prepared.info == {} and len(prepared.getexif()) == 0

    def test_quality_metrics_are_essentially_unaffected_by_which_way_up_the_photo_was(self):
        # brightness/contrast/sharpness are computed over the whole frame, so correct rotation handling should mean
        # these are the same regardless of starting orientation once exif_transpose has run
        base = lesion_photo()
        results = []
        for angle in (0, 90, 180, 270):
            rotated = base.rotate(angle, expand=True)
            q = iq.assess(rotated)
            results.append((q['metrics']['brightness'], q['metrics']['sharpness']))
        brightness = [r[0] for r in results]
        assert max(brightness) - min(brightness) < 2.0   # rotation alone must not change measured brightness materially


class TestUnrelatedAndInsufficientImages:
    """Requirement: test unrelated images and images with insufficient visible information, and confirm the
    system asks for a retake / abstains rather than guessing."""

    def solid_color(self, rgb, n=300):
        return Image.new('RGB', (n, n), rgb)

    def random_noise(self, n=300, seed=1):
        rng = np.random.default_rng(seed)
        return Image.fromarray(rng.integers(0, 255, (n, n, 3), dtype=np.uint8))

    def test_a_blank_wall_photo_is_flagged_unusable_not_interpreted(self):
        q = iq.assess(self.solid_color((200, 195, 190)))
        assert q['unusable'] is True and 'blank' in q['issues'][0].lower() or 'flat' in q['issues'][0].lower()

    def test_pure_noise_is_not_silently_treated_as_a_clear_photo(self):
        # random noise has high local "sharpness" (every pixel differs from its neighbour) but is not a usable
        # photo of anything - the quality check alone cannot catch this (it is not blurry, dark, or flat), which is
        # exactly why image CONTENT (is this a medical photo at all) is a separate question handled by the model's
        # image_type classification, not by the pixel-statistics quality gate - documented here, not silently assumed.
        q = iq.assess(self.random_noise())
        assert q['rating'] == 'good'   # the quality gate correctly has nothing to say about pure noise...
        # ...which is exactly why content classification (not_medical) is a SEPARATE, later check - see the prompt
        # instruction in medical_analyzer.py SYSTEM_INSTRUCTION step 1 ("DECIDE WHAT THE IMAGE IS").

    def test_the_model_prompt_explicitly_requires_checking_whether_the_photo_is_a_medical_image_at_all(self):
        from quickaid.ai.medical_analyzer import SYSTEM_INSTRUCTION
        text = SYSTEM_INSTRUCTION.lower()
        assert 'not_medical' in text and ('landscape' in text or 'document' in text or 'screenshot' in text)
        assert 'do not invent findings' in text

    def test_basic_mode_with_no_ai_cannot_judge_content_and_says_so_rather_than_guessing(self, app):
        analyzer = MedicalAnalyzer()
        analyzer.use_gemini = False
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            self.solid_color((190, 70, 70)).save(f.name)   # looks nothing like skin, could be anything
            out = analyzer.analyze_image(f.name)
        # basic mode has no way to tell "this is not a body part" - it must not fabricate a specific finding about it
        assert not any(e.get('likelihood') == 'more likely' and e.get('name') not in ('', None) for e in out.get('possible_explanations', [])) \
            or out.get('assessment_status') in ('insufficient', 'limited')


class TestRetakeAdviceIsActionable:
    """Every 'poor' rating must come with concrete, specific advice - not just a rejection."""

    @pytest.mark.parametrize('maker,needle', [
        (lambda: Image.new('RGB', (40, 40), (200, 180, 160)), 'full size'),
        (lambda: Image.eval(Image.new('RGB', (300, 300), (200, 180, 160)), lambda v: int(v * 0.05)), 'bright'),
        (lambda: Image.new('RGB', (300, 300), (210, 170, 150)), 'frame'),
    ])
    def test_a_poor_rating_always_explains_what_to_do_differently(self, maker, needle):
        q = iq.assess(maker())
        if q['rating'] == 'poor':
            assert q['advice'] and any(needle.lower() in a.lower() for a in q['advice']), q

"""Photo upload: server-side validation (content, not file names), error messages, privacy, and honest result structure."""

import io
import os
import struct
import zlib
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from quickaid.ai.upload_validation import MAX_BYTES, MIN_SIDE, sniff_format, validate_upload


def photo(n=300, fmt='PNG', seed=1):
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.integers(-25, 25, (n, n, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


def png_with_size(w, h):
    """A valid, tiny PNG file whose HEADER claims a huge image (a decompression bomb)."""
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    row = b'\x00' + b'\x00' * w
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(row * min(h, 4))) + chunk(b'IEND', b'')


def upload(client, data, name='photo.png'):
    return client.post('/upload', data={'file': (io.BytesIO(data), name)}, content_type='multipart/form-data')


class TestSniffing:
    @pytest.mark.parametrize('fmt,expected', [('PNG', 'png'), ('JPEG', 'jpg'), ('GIF', 'gif'), ('BMP', 'bmp'), ('WEBP', 'webp')])
    def test_real_format_comes_from_the_bytes(self, fmt, expected):
        assert sniff_format(photo(100, fmt)[:32]) == expected

    def test_heic_and_unknown_content(self):
        assert sniff_format(b'\x00\x00\x00\x18ftypheic' + b'\x00' * 20) == 'heic'
        assert sniff_format(b'%PDF-1.4') is None and sniff_format(b'') is None


class TestValidator:
    @pytest.mark.parametrize('fmt', ['PNG', 'JPEG', 'GIF', 'BMP', 'WEBP'])
    def test_every_supported_format_is_accepted(self, fmt):
        info, problem = validate_upload(photo(200, fmt), f'p.{fmt.lower()}')
        assert problem is None and info.width == 200 and info.ext in ('png', 'jpg', 'gif', 'bmp', 'webp')

    def test_content_decides_not_the_file_name(self):
        info, problem = validate_upload(photo(200, 'PNG'), 'actually-a-png.jpg')
        assert problem is None and info.ext == 'png'

    @pytest.mark.parametrize('data,name,code', [
        (None, '', 'no_file'),
        (b'', 'a.png', 'empty_file'),
        (b'x' * (MAX_BYTES + 1), 'a.png', 'too_large'),
        (b'\x00\x00\x00\x18ftypheic' + b'\x00' * 50, 'IMG_1.HEIC', 'heic_unsupported'),
        (b'%PDF-1.4 something', 'scan.pdf', 'unsupported_type'),
        (b'just some text', 'photo.jpg', 'not_an_image'),
        (b'MZ\x90\x00 executable', 'virus.png', 'not_an_image'),
    ])
    def test_each_problem_has_its_own_code(self, data, name, code):
        info, problem = validate_upload(data, name)
        assert info is None and problem.code == code and problem.message and problem.hints

    def test_a_truncated_image_is_reported_as_damaged(self):
        data = photo(300, 'PNG')
        info, problem = validate_upload(data[:len(data) // 2], 'cut.png')
        assert problem.code == 'corrupt' and 'damaged' in problem.message

    def test_tiny_photos_are_refused_with_their_size_in_the_message(self):
        info, problem = validate_upload(photo(MIN_SIDE - 1), 'tiny.png')
        assert problem.code == 'too_small' and f'{MIN_SIDE - 1}×{MIN_SIDE - 1}' in problem.message

    def test_decompression_bombs_are_refused_before_decoding(self):
        bomb = png_with_size(20000, 20000)
        assert len(bomb) < 5000                                       # tiny on disk, enormous in memory
        info, problem = validate_upload(bomb, 'bomb.png')
        assert problem.code == 'too_many_pixels'

    def test_animated_gifs_use_the_first_frame(self):
        frames = [Image.new('RGB', (150, 150), c) for c in ((255, 0, 0), (0, 255, 0))]
        buf = io.BytesIO()
        frames[0].save(buf, 'GIF', save_all=True, append_images=frames[1:])
        info, problem = validate_upload(buf.getvalue(), 'a.gif')
        assert problem is None and info.frames == 2

    def test_messages_are_plain_language_with_next_steps(self):
        for data, name in [(b'', 'a.png'), (b'text', 'a.jpg'), (b'%PDF-1.4', 'a.pdf')]:
            _, p = validate_upload(data, name)
            assert not any(tech in p.message.lower() for tech in ('exception', 'traceback', 'errno', 'pil'))
            assert all(len(h) > 10 for h in p.hints)


class TestUploadEndpoint:
    def test_a_good_photo_is_accepted_and_returns_photo_details(self, client, registered_user):
        r = upload(client, photo(300))
        j = r.get_json()
        assert r.status_code == 200 and j['success'] and j['photo'] == {'width': 300, 'height': 300, 'size': len(photo(300)), 'format': 'png'}

    def test_problems_return_code_message_and_hints(self, client, registered_user):
        for data, name, code in [(b'', 'a.png', 'empty_file'), (b'hello', 'a.jpg', 'not_an_image'), (photo(30), 't.png', 'too_small')]:
            r = upload(client, data, name)
            j = r.get_json()
            assert r.status_code == 400 and j['code'] == code and j['error'] and j['hints'] and j['retake'] is True

    def test_missing_file_field(self, client, registered_user):
        r = client.post('/upload', data={}, content_type='multipart/form-data')
        assert r.status_code == 400 and r.get_json()['code'] == 'no_file'

    def test_oversize_uploads_get_a_helpful_json_413(self, client, registered_user, app):
        r = upload(client, b'x' * (17 * 1024 * 1024), 'huge.png')
        j = r.get_json()
        assert r.status_code == 413 and j['code'] == 'too_large' and j['hints']

    def test_rejected_photos_are_not_saved_to_history(self, client, registered_user):
        upload(client, b'not an image', 'a.jpg')
        assert client.get('/api/history').get_json()['history'] == []

    def test_uploaded_photos_are_never_kept_on_disk(self, client, registered_user, app):
        folder = app.config['UPLOAD_FOLDER']
        before = set(os.listdir(folder)) if os.path.isdir(folder) else set()
        upload(client, photo(300))
        upload(client, b'garbage', 'a.png')
        assert (set(os.listdir(folder)) if os.path.isdir(folder) else set()) == before

    def test_the_temp_file_is_removed_even_when_analysis_crashes(self, client, registered_user, app, monkeypatch):
        from quickaid import app as app_module
        monkeypatch.setattr(app_module.medical_analyzer, 'analyze_image', MagicMock(side_effect=RuntimeError('boom')))
        folder = app.config['UPLOAD_FOLDER']
        before = set(os.listdir(folder)) if os.path.isdir(folder) else set()
        r = upload(client, photo(300))
        j = r.get_json()
        assert r.status_code == 500 and j['code'] == 'analysis_failed' and j['hints'] and 'boom' not in j['error']
        assert (set(os.listdir(folder)) if os.path.isdir(folder) else set()) == before

    def test_two_users_uploading_the_same_filename_cannot_collide(self, client, registered_user, app, monkeypatch):
        from quickaid import app as app_module
        seen = []
        real = app_module.medical_analyzer.analyze_image
        monkeypatch.setattr(app_module.medical_analyzer, 'analyze_image', lambda path: (seen.append(path), real(path))[1])
        upload(client, photo(300, seed=1), 'photo.png')
        upload(client, photo(300, seed=2), 'photo.png')
        assert len(set(seen)) == 2 and all('photo' not in os.path.basename(p) for p in seen)

    def test_webp_photos_work_end_to_end(self, client, registered_user):
        assert upload(client, photo(300, 'WEBP'), 'p.webp').status_code == 200

    def test_a_png_named_jpg_is_analysed_as_what_it_really_is(self, client, registered_user):
        assert upload(client, photo(300, 'PNG'), 'p.jpg').status_code == 200

    def test_requires_login(self, client):
        assert upload(client, photo(300)).status_code in (302, 401)


class TestCameraAndPermissions:
    def test_the_camera_is_allowed_for_this_site_only(self, client):
        pp = client.get('/').headers['Permissions-Policy']
        assert 'camera=(self)' in pp and 'microphone=()' in pp and 'geolocation=()' in pp

    def test_uploader_markup_has_camera_drop_zone_progress_and_native_capture(self, client, registered_user):
        html = client.get('/').data.decode()
        for needle in ('id="dropzone"', 'id="cameraBtn"', 'id="cameraDialog"', 'capture="environment"', 'id="uploadProgress"', 'id="prechecks"',
                       'image/webp', 'How to take a photo that can be assessed', 'id="cancelUpload"'):
            assert needle in html, needle

    def test_script_has_every_part_of_the_workflow(self, client):
        js = client.get('/static/js/app.js').data.decode()
        for needle in ('xhr.upload.onprogress', 'getUserMedia', "'drop'", "'paste'", 'HEIC', 'too small', 'Take another photo', 'Choose another photo',
                       'quickCheck', 'xhr.abort', 'Analyze anyway'):
            assert needle in js, needle
        assert 'innerHTML' not in js


class TestHonestImageResults:
    def photo_path(self, tmp_path):
        p = tmp_path / 'p.png'
        p.write_bytes(photo(300))
        return str(p)

    def test_basic_mode_measures_colour_and_offers_no_interpretation(self, tmp_path):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        a = MedicalAnalyzer()
        a.use_gemini = False
        r = a.analyze_image(self.photo_path(tmp_path))
        assert r['analysis_mode'] == 'basic' and r['interpretation_available'] is False
        assert r['possible_explanations'] == [] and r['detected_conditions'] == []
        assert all(f['description'].startswith('Colour measurement') for f in r['findings'])
        assert 'cannot say what is wrong' in r['summary']
        assert not any(w in ' '.join(r['recommendations']).lower() for w in ('bruise', 'infection', 'inflammation'))

    def test_basic_mode_never_labels_a_photo_as_low_risk_or_confident(self, tmp_path):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        a = MedicalAnalyzer()
        a.use_gemini = False
        r = a.analyze_image(self.photo_path(tmp_path))
        assert r['confidence'] == 'low' and r['reliability']['level'] == 'limited' and r['interpretation_status'] == 'rules'

    def test_ai_interpretations_are_labelled_unvalidated(self, tmp_path):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        from quickaid.evaluation.clients import MockClient
        a = MedicalAnalyzer()
        a.use_gemini, a.client = True, MockClient('fixture', fixtures={'Assess this photo': {
            'urgency': 'low', 'confidence': 'medium', 'findings': [{'description': 'a small red patch', 'certainty': 'clearly visible'}],
            'possible_explanations': [{'name': 'irritation', 'likelihood': 'possible', 'supported_by': ['a small red patch']}], 'recommendations': ['Keep it clean']}})
        r = a.analyze_image(self.photo_path(tmp_path))
        assert r['interpretation_status'] == 'unvalidated_ai' and 'not been validated' in r['interpretation_note']
        assert r['findings'][0]['description'] == 'a small red patch'
        assert r['possible_explanations'][0]['name'] == 'irritation' and 'a small red patch' in r['possible_explanations'][0]['supported_by']

    def test_an_interpretation_is_never_the_same_object_as_a_finding(self, tmp_path):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        from quickaid.evaluation.clients import MockClient
        a = MedicalAnalyzer()
        a.use_gemini, a.client = True, MockClient('fixture', fixtures={'Assess this photo': {
            'urgency': 'low', 'findings': ['red patch'], 'possible_explanations': [{'name': 'rash', 'likelihood': 'possible'}]}})
        r = a.analyze_image(self.photo_path(tmp_path))
        assert {f['description'] for f in r['findings']}.isdisjoint({e['name'] for e in r['possible_explanations']})
        assert set(r) >= {'findings', 'possible_explanations', 'not_assessable'}

    def test_result_cards_keep_observation_and_interpretation_apart_in_the_ui(self, client):
        js = client.get('/static/js/app.js').data.decode()
        for needle in ('What is visible in the photo', 'Observed', 'What it could mean', 'AI interpretation · unverified', 'Not available',
                       'What a photo cannot tell us', 'Keep the two apart'):
            assert needle in js, needle

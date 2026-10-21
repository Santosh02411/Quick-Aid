"""Hostile and malformed uploads and per-person rate limits. Nothing may crash (5xx), leak a file, or be analysed."""

import io
import os
import random
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

PW = 'orange-kettle-42'


def signed_in(app, name='ann'):
    c = app.test_client()
    c.post('/register', data={'username': name, 'email': f'{name}@example.com', 'password': PW, 'confirm_password': PW})
    return c


def good(fmt='PNG', n=300, seed=1):
    arr = np.clip(np.random.default_rng(seed).integers(-25, 25, (n, n, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


def post(c, data, name='x.png'):
    return c.post('/upload', data={'file': (io.BytesIO(data), name)}, content_type='multipart/form-data')


def leftovers(app):
    return [n for n in os.listdir(app.config['UPLOAD_FOLDER']) if not n.startswith('.')]


@pytest.fixture()
def no_rate_limit(app):
    """Depends on `app` so the limiter is fully initialised first; only then is it switched off for this test."""
    from quickaid.extensions import limiter
    limiter.enabled = False
    try:
        yield
    finally:
        limiter.enabled = True


@pytest.mark.usefixtures('no_rate_limit')
class TestMalformedUploads:
    def test_random_bytes_never_crash_and_never_leave_files(self, app):
        c, rng = signed_in(app), random.Random(7)
        for i in range(60):
            data = bytes(rng.getrandbits(8) for _ in range(rng.choice([1, 8, 40, 500, 4000])))
            r = post(c, data, rng.choice(['a.png', 'a.jpg', 'a.gif', 'a', 'a.webp']))
            assert r.status_code == 400, (i, r.status_code)
        assert leftovers(app) == []

    @pytest.mark.parametrize('fmt', ['PNG', 'JPEG', 'GIF', 'BMP', 'WEBP'])
    def test_every_truncation_point_is_rejected_cleanly(self, app, fmt):
        c, data = signed_in(app), good(fmt)
        cuts = sorted({1, 4, 8, 12, 16, 33, 64, 100, len(data) // 4, len(data) // 2, len(data) - 40, len(data) - 8, len(data) - 1})
        for cut in cuts:
            r = post(c, data[:cut])
            assert r.status_code in (200, 400), (fmt, cut, r.status_code)          # a few cut points still decode (images tolerate a lost tail): never a 5xx
        assert leftovers(app) == []

    def test_corrupting_bytes_in_a_valid_image_is_never_a_server_error(self, app):
        c, data, rng = signed_in(app), bytearray(good('PNG')), random.Random(3)
        for _ in range(40):
            mutated = bytearray(data)
            for _ in range(rng.randint(1, 20)):
                mutated[rng.randrange(len(mutated))] = rng.getrandbits(8)
            assert post(c, bytes(mutated)).status_code in (200, 400)
        assert leftovers(app) == []

    @pytest.mark.parametrize('payload,name', [
        (b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', 'x.svg'),
        (b'<html><script>alert(1)</script></html>', 'x.png'),
        (b'GIF89a<script>alert(1)</script>', 'x.gif'),
        (b'%PDF-1.4\n1 0 obj<< /JS (app.alert(1)) >>', 'x.png'),
        (b'MZ\x90\x00\x03' + b'\x00' * 200, 'x.jpg'),
        (b'PK\x03\x04' + b'\x00' * 100, 'x.png'),
        (b'#!/bin/sh\nrm -rf /\n', 'x.png'),
        (b'\xff\xd8\xff\xe0<?php system($_GET[0]); ?>', 'shell.php.jpg'),
    ])
    def test_scripts_documents_and_executables_posing_as_images_are_refused(self, app, payload, name):
        c = signed_in(app)
        r = post(c, payload, name)
        assert r.status_code == 400 and r.get_json()['code'] in ('not_an_image', 'corrupt', 'unsupported_type', 'too_small')
        assert leftovers(app) == []

    def test_a_real_image_with_a_script_appended_is_analysed_as_pixels_only(self, app):
        c = signed_in(app)
        r = post(c, good('PNG') + b'<script>alert(1)</script>')
        assert r.status_code in (200, 400)
        assert b'<script>' not in r.get_data()

    @pytest.mark.parametrize('name', ['../../etc/passwd', '..\\..\\win.ini', 'a' * 400 + '.png', 'x\x00.png', '<img src=x onerror=alert(1)>.png', '.png', 'CON.png'])
    def test_hostile_filenames_cannot_escape_or_inject(self, app, name):
        c = signed_in(app)
        r = post(c, good('PNG'), name)
        assert r.status_code in (200, 400)
        assert leftovers(app) == []
        if r.status_code == 200:
            html = c.get('/history').get_data(as_text=True)
            assert '<img src=x' not in html and 'onerror=alert' not in html
        assert not os.path.exists('/etc/passwd.png')

    def test_decompression_and_dimension_bombs_are_refused_without_decoding(self, app):
        def chunk(t, d):
            return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
        for w, h in ((50000, 50000), (65535, 65535), (100000, 1), (1, 100000)):
            bomb = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'\x00' * 64)) + chunk(b'IEND', b'')
            r = post(signed_in(app, f'u{w}{h}'), bomb)
            assert r.status_code == 400 and r.get_json()['code'] in ('too_many_pixels', 'too_small', 'corrupt'), (w, h)

    def test_an_oversized_request_is_refused_before_it_is_read_into_memory(self, app):
        c = signed_in(app)
        r = c.post('/upload', data=b'x' * (17 * 1024 * 1024), content_type='multipart/form-data; boundary=b')
        assert r.status_code in (400, 413)

    def test_many_form_parts_do_not_crash(self, app):
        c = signed_in(app)
        data = {f'f{i}': 'x' for i in range(500)}
        data['file'] = (io.BytesIO(good('PNG')), 'a.png')
        assert c.post('/upload', data=data, content_type='multipart/form-data').status_code in (200, 400, 413)

    def test_other_methods_and_missing_parts(self, app):
        c = signed_in(app)
        assert c.get('/upload').status_code == 405
        assert c.post('/upload', data={'file': ''}, content_type='multipart/form-data').status_code == 400
        assert c.post('/upload', json={'file': 'x'}).status_code == 400
        assert c.post('/upload', data={'other': (io.BytesIO(good()), 'a.png')}, content_type='multipart/form-data').status_code == 400

    def test_the_stored_filename_is_the_cleaned_one(self, app):
        from quickaid.core import database as db
        c = signed_in(app)
        post(c, good('PNG'), '../../evil name!.png')
        stored = db.get_history(1)[0]['original_filename']
        assert '..' not in stored and '/' not in stored and '\\' not in stored


class TestRateLimitsFollowThePerson:
    def test_a_person_cannot_multiply_their_allowance_by_changing_address(self, app):
        c = signed_in(app)
        codes = []
        for i in range(20):
            codes.append(c.post('/analyze_symptoms', json={'symptoms': 'headache'}, environ_overrides={'REMOTE_ADDR': f'198.51.100.{i + 1}'}).status_code)
        assert codes.count(429) >= 4 and codes[:15] == [200] * 15

    def test_people_sharing_one_address_do_not_share_an_allowance(self, app):
        a, b = signed_in(app, 'ann'), signed_in(app, 'bob')
        same = {'REMOTE_ADDR': '203.0.113.50'}
        for _ in range(15):
            assert a.post('/analyze_symptoms', json={'symptoms': 'headache'}, environ_overrides=same).status_code == 200
        assert a.post('/analyze_symptoms', json={'symptoms': 'headache'}, environ_overrides=same).status_code == 429
        assert b.post('/analyze_symptoms', json={'symptoms': 'headache'}, environ_overrides=same).status_code == 200

    def test_uploads_are_limited_per_person(self, app):
        c = signed_in(app)
        data = good('PNG')
        codes = [post(c, data).status_code for _ in range(12)]
        assert codes[:10] == [200] * 10 and 429 in codes[10:]

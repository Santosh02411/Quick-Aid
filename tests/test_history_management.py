"""History: search, filters, pagination, reports, notes, export, deletion, and strict per-user access control."""

import csv
import io
import json

import pytest


@pytest.fixture()
def two_users(app, client, registered_user):
    """`client` = alice (id 1, already signed in); `bob` = a second, separate signed-in session (id 2)."""
    bob = app.test_client()
    bob.post('/register', data={'username': 'bob', 'email': 'bob@example.com', 'password': 'supersecret123', 'confirm_password': 'supersecret123'})
    return client, bob


def seed(db_module_or_none=None):
    from quickaid.core import database as db
    ids = {}
    for i in range(1, 8):
        ids[f's{i}'] = db.save_symptom_analysis(1, f'headache number {i}', {
            'urgency_level': ['low', 'medium', 'high'][i % 3], 'emergency_alert': {'alert': i == 3}, 'possible_conditions': ['migraine'] if i % 2 else ['tension headache'],
            'recommendations': ['rest in a dark room'], 'detected_symptoms': ['headache'], 'summary': f'summary {i}'})
    ids['img'] = db.save_image_analysis(1, 'forearm-rash.png', {'urgency': 'medium', 'detected_conditions': ['eczema'], 'recommendations': ['moisturise'],
                                                               'findings': [{'description': 'a red patch', 'certainty': 'clearly visible'}]})
    return ids


class TestSearchFilterPaginate:
    def test_default_listing_keeps_the_original_response_shape(self, client, registered_user):
        seed()
        j = client.get('/api/history').get_json()
        assert j['success'] and len(j['history']) == 8 and j['pagination']['total'] == 8 and j['request_id']

    def test_text_search_matches_symptoms_conditions_advice_and_filenames(self, client, registered_user):
        seed()
        for term, expected in [('eczema', 1), ('forearm', 1), ('tension', 3), ('dark room', 7), ('headache number 4', 1), ('MIGRAINE', 4)]:
            assert client.get(f'/api/history?q={term}').get_json()['pagination']['total'] == expected, term

    def test_all_search_words_must_match(self, client, registered_user):
        seed()
        assert client.get('/api/history?q=eczema+headache').get_json()['pagination']['total'] == 0
        assert client.get('/api/history?q=migraine+number+3').get_json()['pagination']['total'] == 1

    def test_filters_by_type_urgency_emergency_and_date(self, client, registered_user):
        seed()
        total = lambda qs: client.get('/api/history?' + qs).get_json()['pagination']['total']   # noqa: E731
        assert total('type=image') == 1 and total('type=symptom') == 7 and total('type=symptom,image') == 8 and total('type=symptom&type=image') == 8
        assert total('urgency=high') == 2 and total('urgency=high,medium') > 2
        assert total('emergency=true') == 1 and total('emergency=false') == 7
        assert total('date_from=2000-01-01') == 8 and total('date_from=2999-01-01') == 0 and total('date_to=2000-01-01') == 0
        assert total('type=symptom&urgency=high&emergency=false') == 2 and total('type=symptom&urgency=low&emergency=true') == 1

    def test_filters_and_search_combine(self, client, registered_user):
        seed()
        j = client.get('/api/history?q=migraine&urgency=high').get_json()
        assert all(e['level'] == 'high' and 'migraine' in e['conditions'] for e in j['history'])

    def test_pagination(self, client, registered_user):
        seed()
        p1 = client.get('/api/history?per_page=3&page=1').get_json()
        p3 = client.get('/api/history?per_page=3&page=3').get_json()
        assert len(p1['history']) == 3 and p1['pagination'] == {'page': 1, 'per_page': 3, 'total': 8, 'pages': 3, 'has_next': True, 'has_prev': False}
        assert len(p3['history']) == 2 and p3['pagination']['has_next'] is False
        seen = [e['id'] for pg in (1, 2, 3) for e in client.get(f'/api/history?per_page=3&page={pg}').get_json()['history']]
        assert len(seen) == 8 and len({(e) for e in seen}) >= 7          # no entry repeated across pages
        assert client.get('/api/history?page=99').get_json()['history'] == []

    def test_pagination_also_works_with_a_text_search(self, client, registered_user):
        seed()
        j = client.get('/api/history?q=headache&per_page=3&page=2').get_json()
        assert len(j['history']) == 3 and j['pagination']['total'] == 7 and j['pagination']['pages'] == 3

    def test_sorting(self, client, registered_user):
        seed()
        newest = [e['created_at'] for e in client.get('/api/history?sort=newest').get_json()['history']]
        oldest = [e['created_at'] for e in client.get('/api/history?sort=oldest').get_json()['history']]
        assert newest == sorted(newest, reverse=True) and oldest == sorted(oldest)
        levels = [e['level'] for e in client.get('/api/history?sort=urgency').get_json()['history']]
        rank = {'high': 0, 'medium': 1, 'low': 2}
        assert [rank[l] for l in levels] == sorted(rank[l] for l in levels)

    @pytest.mark.parametrize('qs,field', [('page=0', 'page'), ('page=abc', 'page'), ('per_page=101', 'per_page'), ('per_page=0', 'per_page'),
                                          ('date_from=31-12-2026', 'date_from'), ('date_to=nonsense', 'date_to'), ('sort=random', 'sort'),
                                          ('urgency=critical', 'urgency.0'), ('type=video', 'type.0'), ('emergency=maybe', 'emergency'),
                                          ('q=' + 'x' * 101, 'q')])
    def test_bad_filters_get_a_structured_400_naming_the_field(self, client, registered_user, qs, field):
        r = client.get('/api/history?' + qs)
        j = r.get_json()
        assert r.status_code == 400 and j['success'] is False and j['code'] == 'validation_error' and j['request_id']
        assert any(d['field'] == field for d in j['details']), j['details']

    def test_the_page_renders_results_filters_and_pagination_links(self, client, registered_user):
        seed()
        html = client.get('/history?q=headache&per_page=3').data.decode()
        assert 'Showing 1–3 of 7 matching' in html and 'value="headache"' in html and 'rel="next"' in html
        assert 'q=headache' in html and 'per_page=3' in html and 'page=2' in html           # filters survive in the page links
        assert '/chat?type=symptom&amp;id=' in html or '/chat?type=symptom&id=' in html

    def test_the_page_survives_bad_input_instead_of_erroring(self, client, registered_user):
        seed()
        r = client.get('/history?page=-4&date_from=junk')
        assert r.status_code == 200 and 'Showing everything instead' in r.data.decode()

    def test_empty_states(self, client, registered_user):
        assert 'No checks yet' in client.get('/history').data.decode()
        seed()
        assert 'No checks match' in client.get('/history?q=zzzzzz').data.decode()


class TestReportsAndNotes:
    def test_detail_returns_the_full_analysis(self, client, registered_user):
        ids = seed()
        e = client.get(f"/api/history/symptom/{ids['s1']}").get_json()['entry']
        assert e['full']['summary'] == 'summary 1' and e['symptom_text'] == 'headache number 1' and e['notes'] == '' and e['conversations'] == []
        img = client.get(f"/api/history/image/{ids['img']}").get_json()['entry']
        assert img['full']['findings'][0]['description'] == 'a red patch'

    def test_report_page_renders_for_both_kinds(self, client, registered_user):
        ids = seed()
        s = client.get(f"/history/symptom/{ids['s1']}").data.decode()
        assert 'headache number 1' in s and 'My notes' in s and 'not a diagnosis' in s and 'Print' in s
        i = client.get(f"/history/image/{ids['img']}").data.decode()
        assert 'What is visible in the photo' in i and 'a red patch' in i and 'Observed' in i

    def test_notes_roundtrip_clear_and_search(self, client, registered_user):
        ids = seed()
        url = f"/api/history/symptom/{ids['s1']}/notes"
        r = client.patch(url, json={'notes': '  Ask the GP about my sleep  '}).get_json()
        assert r['success'] and r['notes'] == 'Ask the GP about my sleep' and r['notes_updated_at']
        assert client.get(f"/api/history/symptom/{ids['s1']}").get_json()['entry']['notes'] == 'Ask the GP about my sleep'
        assert client.get('/api/history?q=sleep').get_json()['pagination']['total'] == 1           # notes are searchable
        assert [e['has_notes'] for e in client.get('/api/history?q=sleep').get_json()['history']] == [True]
        assert client.patch(url, json={'notes': ''}).get_json()['notes'] == ''
        assert client.get('/api/history?q=sleep').get_json()['pagination']['total'] == 0

    def test_notes_validation(self, client, registered_user):
        ids = seed()
        url = f"/api/history/symptom/{ids['s1']}/notes"
        assert client.patch(url, json={'notes': 'x' * 2001}).status_code == 400
        assert client.patch(url, json={'notes': 123}).get_json()['code'] == 'validation_error'
        assert client.patch(url, data='[1,2]', content_type='application/json').get_json()['code'] == 'invalid_json'
        assert client.patch(url, json={'notes': 'x' * 2000}).status_code == 200

    def test_notes_are_encrypted_at_rest(self, client, registered_user):
        from quickaid.core import database as db
        ids = seed()
        client.patch(f"/api/history/symptom/{ids['s1']}/notes", json={'notes': 'a very private thought'})
        with db.get_connection() as conn:
            raw = conn.execute('SELECT notes FROM symptom_analyses WHERE id = ?', (ids['s1'],)).fetchone()[0]
        assert 'private' not in raw and raw.startswith('enc:')

    def test_health_text_is_encrypted_in_every_history_column(self, client, registered_user):
        from quickaid.core import database as db
        seed()
        with db.get_connection() as conn:
            row = conn.execute('SELECT symptom_text, detected_symptoms, possible_conditions, recommendations, safety_tips FROM symptom_analyses').fetchone()
            img = conn.execute('SELECT original_filename, detected_conditions, recommendations FROM image_analyses').fetchone()
        assert all(v.startswith('enc:') for v in list(row) + list(img))
        assert 'headache' not in ' '.join(row) and 'forearm' not in ' '.join(img)

    def test_linked_conversations_appear_on_the_report(self, client, registered_user):
        ids = seed()
        client.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': ids['s1'], 'question': 'Is it serious?'})
        e = client.get(f"/api/history/symptom/{ids['s1']}").get_json()['entry']
        assert len(e['conversations']) == 1 and e['conversations'][0]['message_count'] == 2


class TestDeletion:
    def test_delete_one_entry_and_its_chats(self, client, registered_user):
        ids = seed()
        client.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': ids['s1'], 'question': 'Is it serious?'})
        client.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': ids['s2'], 'question': 'And this one?'})
        assert client.delete(f"/api/history/symptom/{ids['s1']}").get_json()['success']
        assert client.get(f"/api/history/symptom/{ids['s1']}").status_code == 404
        assert client.get('/api/history').get_json()['pagination']['total'] == 7
        chats = client.get('/api/chats').get_json()['conversations']
        assert len(chats) == 1                                           # only the chat about s2 is left

    def test_deleting_twice_is_a_404_not_a_crash(self, client, registered_user):
        ids = seed()
        assert client.delete(f"/api/history/symptom/{ids['s1']}").status_code == 200
        assert client.delete(f"/api/history/symptom/{ids['s1']}").status_code == 404

    def test_unknown_kinds_and_ids_are_404(self, client, registered_user):
        for url in ('/api/history/video/1', '/api/history/symptom/999999', '/api/history/symptom/abc'):
            assert client.get(url).status_code == 404
        assert client.get('/history/symptom/999999').status_code == 404

    def test_clear_all(self, client, registered_user):
        seed()
        assert client.post('/api/history/clear').get_json()['success']
        assert client.get('/api/history').get_json()['pagination']['total'] == 0


class TestExport:
    def test_json_export_has_everything_including_notes_and_chats(self, client, registered_user):
        ids = seed()
        client.patch(f"/api/history/symptom/{ids['s1']}/notes", json={'notes': 'my note'})
        client.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': ids['s1'], 'question': 'Is it serious?'})
        r = client.get('/api/history/export?format=json')
        assert r.status_code == 200 and 'attachment; filename="quickaid-export-' in r.headers['Content-Disposition'] and r.mimetype == 'application/json'
        d = json.loads(r.data)
        assert len(d['assessments']) == 8 and d['account']['username'] == 'testuser' and 'password' not in json.dumps(d).lower()
        s1 = [a for a in d['assessments'] if a['id'] == ids['s1'] and a['type'] == 'symptom'][0]
        assert s1['notes'] == 'my note' and s1['full']['summary'] == 'summary 1'
        assert d['conversations'][0]['messages'][0]['content'] == 'Is it serious?'

    def test_csv_export(self, client, registered_user):
        seed()
        r = client.get('/api/history/export?format=csv')
        rows = list(csv.reader(io.StringIO(r.data.decode('utf-8-sig'))))
        assert r.mimetype == 'text/csv' and rows[0][:4] == ['type', 'id', 'created_at', 'urgency'] and len(rows) == 9

    def test_csv_cannot_smuggle_spreadsheet_formulas(self, client, registered_user):
        from quickaid.core import database as db
        db.save_symptom_analysis(1, '=HYPERLINK("http://evil.example","click")', {'urgency_level': 'low', 'possible_conditions': ['+cmd|calc']})
        rows = list(csv.reader(io.StringIO(client.get('/api/history/export?format=csv').data.decode('utf-8-sig'))))
        assert rows[1][5].startswith("'=") and rows[1][6].startswith("'+")

    def test_single_entry_export(self, client, registered_user):
        ids = seed()
        r = client.get(f"/api/history/symptom/{ids['s1']}/export")
        assert 'attachment' in r.headers['Content-Disposition'] and json.loads(r.data)['entry']['id'] == ids['s1']

    def test_export_format_is_validated(self, client, registered_user):
        assert client.get('/api/history/export?format=xml').status_code == 400

    def test_export_is_rate_limited(self, client, registered_user):
        codes = [client.get('/api/history/export').status_code for _ in range(12)]
        assert codes[:10] == [200] * 10 and 429 in codes[10:]


class TestAccessControl:
    """Bob must never see, change or delete anything of Alice's, and must not be able to tell it exists."""

    def test_the_whole_matrix(self, two_users):
        alice, bob = two_users
        ids = seed()
        alice.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': ids['s1'], 'question': 'private question'})
        chat_id = alice.get('/api/chats').get_json()['conversations'][0]['id']
        s, i = ids['s1'], ids['img']

        not_found = [
            bob.get(f'/api/history/symptom/{s}'), bob.get(f'/api/history/image/{i}'), bob.get(f'/history/symptom/{s}'),
            bob.get(f'/api/history/symptom/{s}/export'), bob.patch(f'/api/history/symptom/{s}/notes', json={'notes': 'hacked'}),
            bob.delete(f'/api/history/symptom/{s}'), bob.delete(f'/api/history/image/{i}'),
            bob.get(f'/api/chats/{chat_id}'), bob.post(f'/api/chats/{chat_id}/messages', json={'message': 'hi'}),
            bob.patch(f'/api/chats/{chat_id}', json={'title': 'mine now'}), bob.get(f'/api/chats/{chat_id}/export'), bob.delete(f'/api/chats/{chat_id}'),
            bob.post('/api/chats', json={'analysis_type': 'symptom', 'analysis_id': s}),
            bob.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': s, 'question': 'tell me about it'}),
        ]
        assert [r.status_code for r in not_found] == [404] * len(not_found)
        # ...and exactly the same answer as for an id that never existed, so nothing is revealed by probing
        assert bob.get('/api/history/symptom/999999').get_json()['error'] == bob.get(f'/api/history/symptom/{s}').get_json()['error']

        assert bob.get('/api/history').get_json()['history'] == [] and bob.get('/api/history?q=headache').get_json()['history'] == []
        assert bob.get('/api/chats').get_json()['conversations'] == [] and bob.get('/api/analyses/recent').get_json()['analyses'] == []
        assert json.loads(bob.get('/api/history/export').data)['assessments'] == []
        assert 'headache' not in bob.get('/history').data.decode()

        # Alice's data is untouched by all of that
        e = alice.get(f'/api/history/symptom/{s}').get_json()['entry']
        assert e['notes'] == '' and alice.get('/api/history').get_json()['pagination']['total'] == 8
        assert len(alice.get(f'/api/chats/{chat_id}').get_json()['messages']) == 2

    def test_bobs_clear_and_retention_only_affect_bob(self, two_users):
        alice, bob = two_users
        seed()
        bob.post('/api/history/clear')
        bob.put('/api/privacy/retention', json={'days': 30})
        assert alice.get('/api/history').get_json()['pagination']['total'] == 8
        assert alice.get('/api/privacy/retention').get_json()['retention']['chosen_days'] is None

    def test_every_data_route_requires_login(self, app):
        anon = app.test_client()
        for method, url in [('get', '/api/history'), ('get', '/api/history/export'), ('get', '/api/history/symptom/1'), ('patch', '/api/history/symptom/1/notes'),
                            ('delete', '/api/history/symptom/1'), ('post', '/api/history/clear'), ('get', '/api/chats'), ('get', '/api/chats/1/export'),
                            ('patch', '/api/chats/1'), ('get', '/api/privacy/retention'), ('put', '/api/privacy/retention')]:
            r = getattr(anon, method)(url)
            assert r.status_code == 401 and r.get_json()['code'] == 'unauthenticated', (method, url)
        for url in ('/history', '/history/symptom/1', '/chat'):
            assert anon.get(url).status_code == 302

    def test_responses_with_medical_data_are_never_cached(self, client, registered_user):
        ids = seed()
        for url in ('/api/history', '/history', f"/history/symptom/{ids['s1']}", '/api/history/export', '/chat', '/account', f"/api/history/symptom/{ids['s1']}"):
            cc = client.get(url).headers.get('Cache-Control', '')
            assert 'no-store' in cc, url

    def test_a_deleted_account_leaves_nothing_behind(self, client, registered_user):
        from quickaid.core import database as db
        seed()
        r = client.post('/account/delete', data={'current_password': registered_user['password'], 'confirm_text': 'DELETE'})
        assert r.status_code == 302
        with db.get_connection() as conn:
            for t in ('symptom_analyses', 'image_analyses', 'chat_conversations', 'chat_messages'):
                assert conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] == 0

/* Quick Aid - history list, report page and privacy controls. Server data goes in via textContent only. */
(function () {
  'use strict';
  var doc = document;
  var CSRF = (doc.querySelector('meta[name="csrf-token"]') || {}).content || '';

  function api(method, url, body) {
    var opts = { method: method, headers: { 'X-CSRFToken': CSRF } };
    if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    return fetch(url, opts).then(function (r) {
      if (r.status === 401) { window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname); return Promise.reject(new Error('auth')); }
      return r.json().catch(function () { return { success: false, error: 'Unexpected server response.' }; });
    });
  }

  /* ---- delete one entry (list cards and the report page) ---- */
  doc.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-delete-entry]');
    if (btn) {
      var parts = btn.getAttribute('data-delete-entry').split(':');
      if (!window.confirm('Delete this check and any chats about it? This cannot be undone.')) return;
      btn.disabled = true;
      api('DELETE', '/api/history/' + encodeURIComponent(parts[0]) + '/' + encodeURIComponent(parts[1])).then(function (d) {
        if (!d.success) { btn.disabled = false; window.alert(d.error || 'Could not delete that entry.'); return; }
        var after = btn.getAttribute('data-after');
        if (after) { window.location.href = after; return; }
        window.location.reload();      // keeps the counts, the page numbers and the next entries correct
      }).catch(function () { btn.disabled = false; });
      return;
    }
    if (e.target.closest('[data-print]')) window.print();
  });

  /* ---- personal notes ---- */
  var form = doc.querySelector('[data-notes-form]');
  if (form) {
    var area = doc.getElementById('notesText'), count = doc.getElementById('notesCount'), msg = doc.getElementById('notesMsg');
    var parts = form.getAttribute('data-entry').split(':');
    area.addEventListener('input', function () { count.textContent = String(area.value.length); msg.textContent = ''; msg.className = 'match-msg'; });
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var submit = form.querySelector('button[type="submit"]'); submit.disabled = true; msg.className = 'match-msg'; msg.textContent = 'Saving…';
      api('PATCH', '/api/history/' + parts[0] + '/' + parts[1] + '/notes', { notes: area.value }).then(function (d) {
        if (d.success) { msg.className = 'match-msg ok'; msg.textContent = d.notes ? 'Saved.' : 'Note cleared.'; }
        else { msg.className = 'match-msg bad'; msg.textContent = d.error || 'Could not save.'; }
      }).catch(function () { msg.className = 'match-msg bad'; msg.textContent = 'Network problem. Try again.'; })
        .then(function () { submit.disabled = false; });
    });
  }

  /* ---- retention choice (account page) ---- */
  var sel = doc.querySelector('[data-retention-select]');
  if (sel) {
    sel.addEventListener('change', function () {
      var msgEl = doc.getElementById('retentionMsg'), v = sel.value;
      api('PUT', '/api/privacy/retention', { days: v === 'default' ? null : parseInt(v, 10) }).then(function (d) {
        msgEl.className = 'match-msg ' + (d.success ? 'ok' : 'bad');
        msgEl.textContent = d.success ? (d.retention.effective_days === 0 ? 'Saved. Your history is kept until you delete it.' : 'Saved. Entries older than ' + d.retention.effective_days + ' days will be deleted automatically.') : (d.error || 'Could not save.');
      }).catch(function () { msgEl.className = 'match-msg bad'; msgEl.textContent = 'Network problem. Try again.'; });
    });
  }
})();

/* Quick Aid - front-end behaviour.
   Everything here is progressive: forms still work without JS (the server
   validates everything again). Result rendering uses textContent only, so
   model output can never inject markup. */
(function () {
  'use strict';

  var doc = document;
  var csrfMeta = doc.querySelector('meta[name="csrf-token"]');
  var CSRF = csrfMeta ? csrfMeta.content : '';

  /* ---------- tiny DOM helper ---------- */
  function h(tag, props) {
    var el = doc.createElement(tag);
    props = props || {};
    Object.keys(props).forEach(function (k) {
      if (k === 'class') el.className = props[k];
      else if (k === 'text') el.textContent = props[k];
      else el.setAttribute(k, props[k]);
    });
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c == null || c === false) continue;
      el.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c);
    }
    return el;
  }

  /* ---------- theme ---------- */
  doc.addEventListener('click', function (e) {
    var t = e.target.closest('[data-theme-toggle]');
    if (!t) return;
    var next = doc.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    doc.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('qa-theme', next); } catch (err) { /* private mode */ }
  });

  /* ---------- show / hide password ---------- */
  doc.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-toggle-pw]');
    if (!btn) return;
    var input = doc.getElementById(btn.getAttribute('data-toggle-pw'));
    if (!input) return;
    var show = input.type === 'password';
    input.type = show ? 'text' : 'password';
    btn.setAttribute('aria-pressed', show ? 'true' : 'false');
    btn.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
    input.focus({ preventScroll: true });
  });

  /* ---------- password strength + confirm match ---------- */
  function scorePassword(pw) {
    if (!pw) return 0;
    var s = 0;
    if (pw.length >= 8) s++;
    if (pw.length >= 12) s++;
    if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) s++;
    if (/\d/.test(pw) && /[^A-Za-z0-9]/.test(pw)) s++;
    if (/^(.)\1+$/.test(pw) || /^(password|12345678|qwertyui)/i.test(pw)) s = Math.min(s, 1);
    return Math.max(1, Math.min(4, s));
  }

  function updateMatch(confirmInput) {
    var otherId = confirmInput.getAttribute('data-match');
    var other = doc.getElementById(otherId);
    var msg = doc.querySelector('[data-match-msg="' + confirmInput.id + '"]');
    if (!other || !msg) return true;
    if (!confirmInput.value) { msg.textContent = ''; msg.className = 'match-msg'; confirmInput.setCustomValidity(''); return true; }
    var ok = confirmInput.value === other.value;
    msg.textContent = ok ? 'Passwords match' : 'Passwords do not match';
    msg.className = 'match-msg ' + (ok ? 'ok' : 'bad');
    confirmInput.setCustomValidity(ok ? '' : 'Passwords do not match');
    return ok;
  }

  doc.addEventListener('input', function (e) {
    var el = e.target;
    if (el.matches && el.matches('[data-strength]')) {
      var bar = doc.querySelector('[data-strength-for="' + el.id + '"]');
      if (bar) bar.setAttribute('data-score', String(scorePassword(el.value)));
    }
    if (el.matches && el.matches('input[type="password"], input[type="text"]')) {
      // re-check any confirm field that points at this input, or this input if it is the confirm field
      doc.querySelectorAll('[data-match]').forEach(function (c) {
        if (c === el || c.getAttribute('data-match') === el.id) updateMatch(c);
      });
    }
  });

  doc.addEventListener('submit', function (e) {
    var form = e.target;
    var msg = form.getAttribute && form.getAttribute('data-confirm');
    if (msg && !window.confirm(msg)) { e.preventDefault(); return; }
    if (form.hasAttribute && form.hasAttribute('data-password-form')) {
      var bad = false;
      form.querySelectorAll('[data-match]').forEach(function (c) { if (!updateMatch(c)) bad = true; });
      if (bad) { e.preventDefault(); form.reportValidity(); }
    }
  });

  /* ---------- greeting by local time ---------- */
  var greet = doc.getElementById('greet');
  if (greet) {
    var hr = new Date().getHours();
    greet.textContent = hr < 5 ? 'Hello' : hr < 12 ? 'Good morning' : hr < 17 ? 'Good afternoon' : hr < 22 ? 'Good evening' : 'Hello';
  }

  /* ---------- history: clear ---------- */
  var clearBtn = doc.getElementById('clearHistoryBtn');
  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      if (!window.confirm('Clear all saved history? This cannot be undone.')) return;
      fetch('/api/history/clear', { method: 'POST', headers: { 'X-CSRFToken': CSRF } })
        .then(function (r) { if (r.ok) window.location.reload(); else window.alert('Could not clear history. Please try again.'); })
        .catch(function () { window.alert('Could not clear history. Please try again.'); });
    });
  }

  /* ======================================================================
     Dashboard tools
     ====================================================================== */
  var emergencyNumber = doc.body.getAttribute('data-emergency') || 'your local emergency number';

  var TRIAGE = {
    low:    { title: 'Likely manageable at home', msg: 'Follow the steps below and keep an eye out for changes.' },
    medium: { title: 'See a doctor soon', msg: 'Arrange a visit within a day or two, or sooner if it gets worse.' },
    high:   { title: 'Get medical help now', msg: null }
  };
  var SVG_NS = 'http://www.w3.org/2000/svg';
  function icon(path) {
    var s = doc.createElementNS(SVG_NS, 'svg');
    s.setAttribute('class', 'icon'); s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('aria-hidden', 'true');
    var p = doc.createElementNS(SVG_NS, 'path'); p.setAttribute('d', path); s.appendChild(p);
    return s;
  }
  var ICON_ALERT = 'm21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3M12 9v4M12 17h.01';
  var ICON_OK = 'M22 11.08V12a10 10 0 1 1-5.93-9.14M22 4 12 14l-3-3';

  function list(items, cls) {
    var ul = h('ul', cls ? { class: cls } : {});
    items.forEach(function (t) { ul.appendChild(h('li', { text: String(t).replace(/^[\u{1F300}-\u{1FAFF}\u2600-\u27BF\uFE0F\s]+/u, '') })); });
    return ul;
  }
  function section(container, title, items, cls) {
    if (!items || !items.length) return;
    container.appendChild(h('h3', { text: title }));
    container.appendChild(list(items, cls));
  }
  function chips(container, title, items) {
    if (!items || !items.length) return;
    container.appendChild(h('h3', { text: title }));
    var ul = h('ul', { class: 'finding-list' });
    items.forEach(function (t) { ul.appendChild(h('li', { text: String(t).replace(/_/g, ' ') })); });
    container.appendChild(ul);
  }

  function bump(id) {
    var el = doc.getElementById(id);
    if (el) el.textContent = String((parseInt(el.textContent, 10) || 0) + 1);
  }
  function countCheck(kind, analysis) {
    bump('statTotal'); bump(kind === 'image' ? 'statImages' : 'statSymptoms');
    var lvl = analysis.urgency || analysis.urgency_level;
    if (lvl === 'high' || (analysis.emergency_alert && analysis.emergency_alert.alert)) {
      bump('statUrgent');
      var card = doc.getElementById('statUrgent');
      if (card) card.parentNode.classList.add('stat-alert');
    }
  }

  function renderError(container, message) {
    container.replaceChildren(h('div', { class: 'alert alert-error', role: 'alert', text: message }));
    container.classList.add('on');
  }

  function renderResult(container, a, kind) {
    var nodes = [];
    var level = a.urgency || a.urgency_level || 'medium';
    if (a.emergency_alert && a.emergency_alert.alert) level = 'high';
    if (!TRIAGE[level]) level = 'medium';
    var t = TRIAGE[level];
    // A rule-based photo check cannot establish that something is minor, so don't claim it.
    var unknown = kind === 'image' && a.analysis_mode === 'basic' && level === 'low';
    if (unknown) t = { title: 'Rough check only', msg: 'Without AI, this check cannot judge how serious it is. If you are unsure, see a doctor.' };

    // honest notices first: basic mode / unusable photo
    if (a.analysis_mode === 'basic') {
      nodes.push(h('div', { class: 'alert alert-warn', text: 'AI is not connected, so this is a limited rule-based check. Treat it as a rough guide only.' }));
    }
    if (a.image_quality === 'poor' || a.image_type === 'not_medical') {
      nodes.push(h('div', { class: 'alert alert-warn', text: a.quality_note || 'This photo is hard to assess. Try a sharper, closer photo in good light.' }));
    }

    var msgBox = h('div', {});
    var lvl = h('p', { class: 'lvl', text: t.title });
    msgBox.appendChild(lvl);
    if (level === 'high') {
      var p = h('p', {});
      p.appendChild(doc.createTextNode('Call '));
      p.appendChild(h('a', { href: 'tel:' + emergencyNumber, text: emergencyNumber }));
      p.appendChild(doc.createTextNode(' or go to the nearest emergency department now.'));
      msgBox.appendChild(p);
      if (a.emergency_alert && a.emergency_alert.reason) msgBox.appendChild(h('p', { text: a.emergency_alert.reason }));
    } else {
      msgBox.appendChild(h('p', { text: t.msg }));
    }
    nodes.push(h('div', { class: 'triage', 'data-level': unknown ? 'unknown' : level }, icon(level === 'low' && !unknown ? ICON_OK : ICON_ALERT), msgBox));

    if (a.summary) nodes.push(h('p', { text: a.summary }));

    var body = h('div', {});
    if (kind === 'image') {
      var meta = h('div', { class: 'meta-row' });
      if (a.confidence) meta.appendChild(h('span', { class: 'badge badge-ok', text: 'Confidence: ' + a.confidence }));
      if (meta.childNodes.length) body.appendChild(meta);
      chips(body, 'What this may be', a.detected_conditions);
      section(body, 'What we can see', a.observations);
    } else {
      chips(body, 'Symptoms noted', a.detected_symptoms);
      chips(body, 'Possible causes', a.possible_conditions);
    }
    section(body, 'Warning signs to watch for', a.red_flags);
    section(body, 'What to do now', a.recommendations);
    section(body, 'Avoid', a.avoid, 'avoid');
    section(body, 'See a doctor if', a.see_doctor_when);
    if (a.safety_tips && a.safety_tips.length) {
      var d = h('details', {}, h('summary', { text: 'General safety tips' }), list(a.safety_tips));
      d.style.marginTop = '1rem';
      body.appendChild(d);
    }
    nodes.push(body);
    nodes.push(h('p', { class: 'fine', text: a.disclaimer || 'For education only. Not a diagnosis. See a healthcare professional.' }));

    container.replaceChildren.apply(container, nodes);
    container.classList.add('on');
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function parseJsonResponse(resp) {
    if (resp.status === 401) { window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname); return Promise.reject(new Error('auth')); }
    if (resp.status === 413) return Promise.resolve({ error: 'That file is larger than 16 MB. Choose a smaller photo.' });
    return resp.json().catch(function () { return { error: 'The server sent an unexpected response. Please try again.' }; });
  }

  /* ----- image tool ----- */
  var dz = doc.getElementById('dropzone');
  if (dz) {
    var input = doc.getElementById('imageInput');
    var analyzeBtn = doc.getElementById('analyzeBtn');
    var preview = doc.getElementById('preview');
    var previewImg = doc.getElementById('previewImg');
    var loading = doc.getElementById('imageLoading');
    var resultBox = doc.getElementById('imageResult');
    var chosen = null;
    var MAX = 16 * 1024 * 1024;

    function choose(file) {
      resultBox.classList.remove('on');
      if (!file) return;
      if (!/^image\/(png|jpe?g|gif|bmp)$/.test(file.type)) { renderError(resultBox, 'Please choose a JPG, PNG, GIF or BMP image.'); return; }
      if (file.size > MAX) { renderError(resultBox, 'That file is larger than 16 MB. Choose a smaller photo.'); return; }
      chosen = file;
      if (previewImg.src) URL.revokeObjectURL(previewImg.src);
      previewImg.src = URL.createObjectURL(file);
      doc.getElementById('previewName').textContent = file.name;
      doc.getElementById('previewSize').textContent = (file.size / 1048576).toFixed(2) + ' MB';
      preview.classList.add('on');
      dz.style.display = 'none';
      analyzeBtn.disabled = false;
    }
    function clearChoice() {
      chosen = null; input.value = '';
      preview.classList.remove('on'); dz.style.display = '';
      analyzeBtn.disabled = true;
    }

    dz.addEventListener('click', function () { input.click(); });
    dz.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
    ['dragenter', 'dragover'].forEach(function (ev) { dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add('dragover'); }); });
    ['dragleave', 'drop'].forEach(function (ev) { dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove('dragover'); }); });
    dz.addEventListener('drop', function (e) { if (e.dataTransfer.files.length) choose(e.dataTransfer.files[0]); });
    input.addEventListener('change', function () { if (input.files.length) choose(input.files[0]); });
    doc.getElementById('removeImage').addEventListener('click', function () { clearChoice(); resultBox.classList.remove('on'); });

    analyzeBtn.addEventListener('click', function () {
      if (!chosen) return;
      var fd = new FormData(); fd.append('file', chosen);
      analyzeBtn.disabled = true; loading.classList.add('on'); resultBox.classList.remove('on');
      fetch('/upload', { method: 'POST', headers: { 'X-CSRFToken': CSRF }, body: fd })
        .then(parseJsonResponse)
        .then(function (data) { if (data.error) renderError(resultBox, data.error); else { renderResult(resultBox, data.analysis, 'image'); countCheck('image', data.analysis); } })
        .catch(function (err) { if (err.message !== 'auth') renderError(resultBox, 'Network problem. Check your connection and try again.'); })
        .then(function () { loading.classList.remove('on'); analyzeBtn.disabled = !chosen; });
    });
  }

  /* ----- symptom tool ----- */
  var symInput = doc.getElementById('symptomsInput');
  if (symInput) {
    var symBtn = doc.getElementById('checkSymptomsBtn');
    var symLoading = doc.getElementById('symptomLoading');
    var symResult = doc.getElementById('symptomResult');
    var count = doc.getElementById('symCount');
    function updateCount() { count.textContent = String(symInput.value.length); }
    symInput.addEventListener('input', updateCount);

    doc.querySelectorAll('[data-chip]').forEach(function (chip) {
      chip.addEventListener('click', function () {
        var word = chip.getAttribute('data-chip');
        var cur = symInput.value.trim();
        if (cur.toLowerCase().indexOf(word) !== -1) return;
        symInput.value = cur ? cur.replace(/[.,;]?$/, ', ') + word : word;
        updateCount(); symInput.focus();
      });
    });

    symBtn.addEventListener('click', function () {
      var text = symInput.value.trim();
      if (text.length < 5) { renderError(symResult, 'Please describe your symptoms in a little more detail.'); return; }
      symBtn.disabled = true; symLoading.classList.add('on'); symResult.classList.remove('on');
      fetch('/analyze_symptoms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF },
        body: JSON.stringify({ symptoms: text })
      })
        .then(parseJsonResponse)
        .then(function (data) { if (data.error) renderError(symResult, data.error); else { renderResult(symResult, data.analysis, 'symptom'); countCheck('symptom', data.analysis); } })
        .catch(function (err) { if (err.message !== 'auth') renderError(symResult, 'Network problem. Check your connection and try again.'); })
        .then(function () { symLoading.classList.remove('on'); symBtn.disabled = false; });
    });
  }
})();

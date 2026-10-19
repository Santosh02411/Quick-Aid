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

  function followUpLink(kind, analysisId) {
    if (!analysisId) return null;
    var link = h('a', { class: 'btn btn-ghost', href: '/chat?type=' + (kind === 'image' ? 'image' : 'symptom') + '&id=' + analysisId });
    link.appendChild(doc.createTextNode('Ask a follow-up question'));
    return h('p', { class: 'result-actions' }, link);
  }

  function explanations(container, items, title) {
    if (!items || !items.length) return false;
    container.appendChild(h('h3', { text: title || 'What it could be' }));
    var ul = h('ul', { class: 'explain' });
    items.forEach(function (e) {
      var tone = e.likelihood === 'more likely' ? 'badge-warn' : 'badge-ok';
      ul.appendChild(h('li', {},
        h('strong', { text: e.name }), doc.createTextNode(' '),
        h('span', { class: 'badge ' + tone, text: e.likelihood }),
        e.why ? h('div', { class: 'muted', text: e.why }) : null,
        (e.supported_by && e.supported_by.length) ? h('div', { class: 'muted', text: 'Based on: ' + e.supported_by.join('; ') }) : null));
    });
    container.appendChild(ul);
    return true;
  }

  function findings(container, items) {
    if (!items || !items.length) return;
    container.appendChild(h('h3', { text: 'What the photo shows' }));
    var ul = h('ul', { class: 'findings' });
    items.forEach(function (f) {
      var d = typeof f === 'string' ? { description: f, certainty: 'possibly visible' } : f;
      ul.appendChild(h('li', {}, doc.createTextNode(d.description + ' '),
        h('span', { class: 'badge ' + (d.certainty === 'clearly visible' ? 'badge-ok' : 'badge-warn'), text: d.certainty })));
    });
    container.appendChild(ul);
  }

  function details(container, title, rows) {
    if (!rows || !rows.length) return;
    container.appendChild(h('h3', { text: title }));
    var dl = h('dl', { class: 'kv' });
    rows.forEach(function (r) { dl.appendChild(h('dt', { text: r.label })); dl.appendChild(h('dd', { text: r.value })); });
    container.appendChild(dl);
  }

  function reliabilityPanel(a) {
    var r = a.reliability;
    if (!r) return null;
    var box = h('div', { class: 'reliability', 'data-level': r.level });
    box.appendChild(h('div', { class: 'rel-head' }, h('strong', { text: 'How reliable is this?' }), h('span', { class: 'rel-label', text: r.label })));
    if (a.uncertainty_statement) box.appendChild(h('p', { class: 'rel-statement', text: a.uncertainty_statement }));
    if (r.reasons && r.reasons.length) {
      var d = h('details', {}, h('summary', { text: 'Why this rating' }), list(r.reasons));
      if (r.ai_self_rating) d.appendChild(h('p', { class: 'hint', text: 'The AI rated its own confidence as "' + r.ai_self_rating.value + '". ' + r.ai_self_rating.note }));
      d.appendChild(h('p', { class: 'hint', text: r.note }));
      box.appendChild(d);
    }
    return box;
  }

  function safetyLog(a) {
    var s = a.safety_review;
    if (!s) return null;
    var items = (s.adjustments && s.adjustments.length) ? list(s.adjustments) : h('p', { class: 'hint', text: 'Our safety checks looked at this result and found nothing to change.' });
    var d = h('details', {}, h('summary', { text: 'Safety checks applied' + (s.urgency_raised ? ' (urgency raised)' : '') }), items);
    d.style.marginTop = '1rem';
    return d;
  }

  function followUpLink(kind, analysisId) {
    if (!analysisId) return null;
    var link = h('a', { class: 'btn btn-ghost', href: '/chat?type=' + (kind === 'image' ? 'image' : 'symptom') + '&id=' + analysisId });
    link.appendChild(doc.createTextNode('Ask a follow-up question'));
    return h('p', { class: 'result-actions' }, link);
  }

  /* Photos: what is VISIBLE (observation) and what it MIGHT MEAN (interpretation) are separate cards on purpose. */
  function rcard(cls, title, badge, badgeTone, note, nodes) {
    var head = h('div', { class: 'rcard-head' }, h('h3', { text: title }), h('span', { class: 'badge ' + badgeTone, text: badge }));
    var card = h('div', { class: 'rcard ' + cls }, head, note ? h('p', { class: 'rcard-note', text: note }) : null);
    nodes.forEach(function (n) { if (n) card.appendChild(n); });
    return card;
  }
  function findingRefs(supported, fs) {
    var out = [];
    (supported || []).forEach(function (txt) {
      var t = String(txt).toLowerCase(), idx = -1;
      fs.forEach(function (f, i) { var d = f.description.toLowerCase(); if (idx < 0 && (d.indexOf(t) >= 0 || t.indexOf(d) >= 0)) idx = i; });
      out.push(idx >= 0 ? 'finding ' + (idx + 1) : txt);
    });
    return out;
  }
  function imageCards(body, a) {
    var fs = (a.findings && a.findings.length ? a.findings : (a.observations || []).map(function (o) { return { description: o, certainty: 'possibly visible' }; }));
    body.appendChild(h('p', { class: 'rcard-intro', text: 'Findings describe what the picture shows. Interpretations are guesses built on those findings. Keep the two apart.' }));

    // 1. observation
    var nodes = [];
    if (fs.length) {
      var ol = h('ol', { class: 'findings numbered' });
      fs.forEach(function (f) {
        ol.appendChild(h('li', {}, doc.createTextNode(f.description + ' '),
          h('span', { class: 'badge ' + (f.certainty === 'clearly visible' ? 'badge-ok' : 'badge-warn'), text: f.certainty })));
      });
      nodes.push(ol);
    } else {
      nodes.push(h('p', { class: 'muted', text: 'Nothing could be reliably described from this photo.' }));
    }
    body.appendChild(rcard('observed', 'What is visible in the photo', 'Observed', 'badge-ok', 'A plain description of what the picture shows. This is not a diagnosis.', nodes));

    // 2. interpretation
    if (a.interpretation_available === false) {
      body.appendChild(rcard('interp unavailable', 'What it could mean', 'Not available', 'badge-warn',
        'Without AI, this check only measures colour. It cannot say what is wrong, so it offers no explanation. Ask a doctor, or connect AI for a (still unverified) interpretation.', []));
    } else if (a.possible_explanations && a.possible_explanations.length) {
      var ul = h('ul', { class: 'explain' });
      a.possible_explanations.forEach(function (e) {
        var refs = findingRefs(e.supported_by, fs);
        ul.appendChild(h('li', {}, h('strong', { text: e.name }), doc.createTextNode(' '),
          h('span', { class: 'badge ' + (e.likelihood === 'more likely' ? 'badge-warn' : 'badge-ok'), text: e.likelihood }),
          e.why ? h('div', { class: 'muted', text: e.why }) : null,
          refs.length ? h('div', { class: 'muted', text: 'Based on: ' + refs.join('; ') }) : null));
      });
      body.appendChild(rcard('interp', 'What it could mean', 'AI interpretation · unverified', 'badge-warn',
        a.interpretation_note || 'These are guesses. They have not been validated against clinical outcomes and may be wrong.', [ul]));
    } else {
      body.appendChild(rcard('interp unavailable', 'What it could mean', 'No interpretation', 'badge-warn',
        'There was not enough clear evidence in the photo to suggest an explanation. That does not mean nothing is wrong.', []));
    }

    // 3. limits
    if (a.not_assessable && a.not_assessable.length) body.appendChild(rcard('limits', 'What a photo cannot tell us', 'Limits', 'badge-ok', '', [list(a.not_assessable)]));
  }

  function renderResult(container, a, kind, analysisId) {
    var nodes = [];
    var level = a.urgency || a.urgency_level || 'medium';
    if (a.emergency_alert && a.emergency_alert.alert) level = 'high';
    if (!TRIAGE[level]) level = 'medium';
    var t = TRIAGE[level];
    var sameDay = false;
    var insufficient = a.assessment_status === 'insufficient' || a.assessment_status === 'invalid';

    // The engine states exactly how soon to be seen.
    if (a.care_level) {
      if (a.care_level === 'emergency') { level = 'high'; t = TRIAGE.high; }
      else if (a.care_level === 'same_day') {
        level = 'high'; sameDay = true;
        t = { title: 'See a doctor today', msg: (a.timeframe && a.timeframe !== 'Today') ? a.timeframe : 'Get checked today, even if it feels manageable.' };
      }
      else if (a.care_level === 'soon') { level = 'medium'; t = { title: 'See a doctor soon', msg: 'Suggested timing: ' + (a.timeframe || 'within 1–2 days') + '.' }; }
      else { level = 'low'; t = { title: 'Likely manageable at home', msg: a.timeframe || TRIAGE.low.msg }; }
    }
    // A rule-based photo check cannot establish that something is minor, so don't claim it.
    var unknown = kind === 'image' && a.analysis_mode === 'basic' && level === 'low';
    if (unknown) t = { title: 'Rough check only', msg: 'Without AI, this check cannot judge how serious it is. If you are unsure, see a doctor.' };
    var emergency = a.emergency_alert && a.emergency_alert.alert;
    if (insufficient && !emergency) {
      unknown = true;
      t = { title: 'Not enough to assess', msg: kind === 'image' ? 'This photo cannot be assessed reliably, so no conclusion is given. See the tips below.' : 'There is not enough information to assess this.' };
    }

    if (a.analysis_mode === 'basic') {
      nodes.push(h('div', { class: 'alert alert-warn', text: 'AI is not connected, so this is only a basic rule-based check. It can miss serious problems, especially if symptoms are described in unusual words or another language. If you are worried, do not wait: call ' + emergencyNumber + ' or see a doctor.' }));
    }
    if (a.quality_check && a.quality_check.rating !== 'good') {
      var qbox = h('div', { class: 'alert alert-warn' }, h('strong', { text: 'Photo quality: ' + a.quality_check.rating + '. ' }), doc.createTextNode((a.quality_check.issues || []).join(' ')));
      if (a.quality_check.advice && a.quality_check.advice.length && !insufficient) qbox.appendChild(list(a.quality_check.advice));
      nodes.push(qbox);
    } else if (a.image_type === 'not_medical') {
      nodes.push(h('div', { class: 'alert alert-warn', text: a.quality_note || 'This does not look like a medical photo. Upload a close, clear photo of the affected area.' }));
    }

    var msgBox = h('div', {});
    msgBox.appendChild(h('p', { class: 'lvl', text: t.title }));
    if (level === 'high' && !sameDay && !insufficient) {
      var p = h('p', {});
      p.appendChild(doc.createTextNode('Call '));
      p.appendChild(h('a', { href: 'tel:' + emergencyNumber, text: emergencyNumber }));
      p.appendChild(doc.createTextNode(' or go to the nearest emergency department now.'));
      msgBox.appendChild(p);
      if (emergency && a.emergency_alert.reason) msgBox.appendChild(h('p', { text: a.emergency_alert.reason }));
    } else {
      msgBox.appendChild(h('p', { text: t.msg }));
      if (sameDay || insufficient) {
        var p2 = h('p', {});
        p2.appendChild(doc.createTextNode(insufficient ? 'If it is painful, bleeding or getting worse, see a doctor, or call ' : 'If it gets worse or you feel unsafe, call '));
        p2.appendChild(h('a', { href: 'tel:' + emergencyNumber, text: emergencyNumber }));
        p2.appendChild(doc.createTextNode(insufficient ? ' in an emergency.' : '.'));
        msgBox.appendChild(p2);
      }
    }
    nodes.push(h('div', { class: 'triage', 'data-level': unknown ? 'unknown' : level }, icon(level === 'low' && !unknown ? ICON_OK : ICON_ALERT), msgBox));

    if (a.summary) nodes.push(h('p', { text: a.summary }));
    var rel = reliabilityPanel(a);
    if (rel) nodes.push(rel);

    var body = h('div', {});
    if (kind === 'image') {
      imageCards(body, a);
    } else {
      section(body, 'Why this level', a.urgency_reasons);
      var rep = a.reported || {};
      chips(body, 'What you reported', (rep.symptoms && rep.symptoms.length) ? rep.symptoms : a.detected_symptoms);
      details(body, 'Details you gave', rep.details);
      if (!explanations(body, a.possible_explanations, 'What it could be')) chips(body, 'Possible causes', a.possible_conditions);
    }
    section(body, 'Warning signs present now', a.red_flags_present, 'present');
    section(body, kind === 'symptom' ? 'What to do' : 'What to do now', a.recommendations);
    section(body, 'Avoid', a.avoid, 'avoid');
    section(body, kind === 'symptom' ? 'Warning signs: get urgent help if' : 'Warning signs to watch for', a.red_flags);
    section(body, 'See a doctor if', a.see_doctor_when);
    section(body, 'What would make this more reliable', a.information_gaps);
    if (a.interview && a.interview.answers && a.interview.answers.length) {
      var qa = h('ul', {});
      a.interview.answers.forEach(function (x) { qa.appendChild(h('li', { text: x.question + ' ' + x.answer })); });
      var dq = h('details', {}, h('summary', { text: 'Your answers (' + a.interview.answers.length + ')' }), qa);
      dq.style.marginTop = '1rem';
      body.appendChild(dq);
    }
    if (a.safety_tips && a.safety_tips.length) {
      var d = h('details', {}, h('summary', { text: 'General safety tips' }), list(a.safety_tips));
      d.style.marginTop = '1rem';
      body.appendChild(d);
    }
    var log = safetyLog(a);
    if (log) body.appendChild(log);
    nodes.push(body);
    nodes.push(h('p', { class: 'fine', text: a.disclaimer || 'For education only. Not a diagnosis. See a healthcare professional.' }));
    var fu = followUpLink(kind, analysisId);
    if (fu) nodes.push(fu);

    container.replaceChildren.apply(container, nodes);
    container.classList.add('on');
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function parseJsonResponse(resp) {
    if (resp.status === 401) { window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname); return Promise.reject(new Error('auth')); }
    if (resp.status === 413) return Promise.resolve({ error: 'That file is larger than 16 MB. Choose a smaller photo.' });
    return resp.json().catch(function () { return { error: 'The server sent an unexpected response. Please try again.' }; });
  }

  /* ----- image tool: choose / drop / paste / camera -> validate -> preview -> upload with progress ----- */
  var dz = doc.getElementById('dropzone');
  if (dz) {
    var input = doc.getElementById('imageInput'), captureInput = doc.getElementById('captureInput');
    var chooseBtn = doc.getElementById('chooseBtn'), cameraBtn = doc.getElementById('cameraBtn');
    var analyzeBtn = doc.getElementById('analyzeBtn'), preview = doc.getElementById('preview'), previewImg = doc.getElementById('previewImg');
    var prechecks = doc.getElementById('prechecks'), precheckNote = doc.getElementById('precheckNote');
    var statusBox = doc.getElementById('uploadStatus'), uploader = doc.getElementById('uploader');
    var progressBox = doc.getElementById('uploadProgress'), bar = doc.getElementById('uploadBar'), stageEl = doc.getElementById('uploadStage');
    var loading = doc.getElementById('imageLoading'), resultBox = doc.getElementById('imageResult');
    var MAX = 16 * 1024 * 1024, MIN_SIDE = 80;
    var ALLOWED = /^image\/(png|jpe?g|gif|bmp|webp)$/, EXT = /\.(png|jpe?g|gif|bmp|webp)$/i;
    var chosen = null, xhr = null, objectUrl = null, weak = false;

    function mb(n) { return (n / 1048576).toFixed(1) + ' MB'; }
    function setStatus(msg) { clearNode(statusBox); if (msg) statusBox.appendChild(h('p', { class: 'hint', text: msg })); }
    function clearNode(el) { while (el.firstChild) el.removeChild(el.firstChild); }

    // A problem with a photo: say what is wrong, say what to do, and offer the next step.
    function showProblem(message, hints, canRetry) {
      clearNode(statusBox);
      var box = h('div', { class: 'alert alert-error', role: 'alert' }, h('strong', { text: message }));
      if (hints && hints.length) box.appendChild(list(hints));
      var row = h('div', { class: 'actions' });
      var again = h('button', { type: 'button', class: 'btn btn-sm' }, canRetry && chosen ? 'Try again' : 'Choose another photo');
      again.addEventListener('click', function () { if (canRetry && chosen) upload(); else input.click(); });
      var cam = h('button', { type: 'button', class: 'btn btn-ghost btn-sm' }, 'Take a photo');
      cam.addEventListener('click', openCameraOrNative);
      row.appendChild(again); row.appendChild(cam); box.appendChild(row);
      statusBox.appendChild(box);
      statusBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function resetPreview() {
      if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
      chosen = null; input.value = ''; captureInput.value = '';
      preview.classList.remove('on'); precheckNote.hidden = true; clearNode(prechecks);
      uploader.hidden = false; analyzeBtn.disabled = true; analyzeBtn.lastChild.textContent = 'Analyze photo'; weak = false;
    }

    /* ---- quick on-device check: lighting, focus, size. A hint for retaking, NOT a medical check. ---- */
    function quickCheck(im) {
      var w = im.naturalWidth, hgt = im.naturalHeight, scale = Math.min(1, 256 / Math.max(w, hgt));
      var cw = Math.max(8, Math.round(w * scale)), ch = Math.max(8, Math.round(hgt * scale));
      var c = doc.createElement('canvas'); c.width = cw; c.height = ch;
      var ctx = c.getContext('2d', { willReadFrequently: true }); ctx.drawImage(im, 0, 0, cw, ch);
      var d = ctx.getImageData(0, 0, cw, ch).data, g = new Float32Array(cw * ch), sum = 0, bright = 0;
      for (var i = 0, j = 0; i < d.length; i += 4, j++) { var l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; g[j] = l; sum += l; if (l > 245) bright++; }
      var mean = sum / g.length, v = 0; for (var k = 0; k < g.length; k++) v += (g[k] - mean) * (g[k] - mean);
      var std = Math.sqrt(v / g.length), lap = 0, n = 0;
      for (var y = 1; y < ch - 1; y++) for (var x = 1; x < cw - 1; x++) {
        var p = y * cw + x, L = -4 * g[p] + g[p - 1] + g[p + 1] + g[p - cw] + g[p + cw]; lap += L * L; n++;
      }
      var sharp = n ? lap / n : 0, glare = bright / g.length, out = [];
      out.push({ s: Math.min(w, hgt) < 240 ? 'warn' : 'ok', t: 'Size ' + w + '×' + hgt + ' px' + (Math.min(w, hgt) < 240 ? ': low resolution, move closer' : '') });
      if (mean < 25) out.push({ s: 'bad', t: 'Lighting: far too dark. Retake in bright light.' });
      else if (mean < 55) out.push({ s: 'warn', t: 'Lighting: quite dark. More light would help.' });
      else if (mean > 240) out.push({ s: 'bad', t: 'Lighting: washed out by too much light.' });
      else if (glare > 0.3) out.push({ s: 'warn', t: 'Lighting: glare on part of the photo.' });
      else out.push({ s: 'ok', t: 'Lighting looks fine' });
      if (std < 6) out.push({ s: 'bad', t: 'Detail: the photo looks blank or one flat colour.' });
      else if (sharp < 3) out.push({ s: 'bad', t: 'Focus: very blurry. Hold steady and tap the screen to focus.' });
      else if (sharp < 8) out.push({ s: 'warn', t: 'Focus: a little soft. Tap the screen to focus.' });
      else out.push({ s: 'ok', t: 'Focus looks fine' });
      return out;
    }

    function decode(file) {
      return new Promise(function (resolve, reject) {
        var url = URL.createObjectURL(file), im = new Image();
        im.onload = function () { resolve({ im: im, url: url }); };
        im.onerror = function () { URL.revokeObjectURL(url); reject(new Error('decode')); };
        im.src = url;
      });
    }

    function choose(file) {
      resultBox.classList.remove('on'); setStatus('');
      if (!file) return;
      var name = file.name || 'photo';
      if (/\.(heic|heif)$/i.test(name) || /image\/hei[cf]/i.test(file.type)) {
        return showProblem("That looks like an iPhone HEIC photo, which can't be read yet.", ['On iPhone: Settings → Camera → Formats → "Most Compatible".', 'Or use the Take a photo button here.']);
      }
      if (!(ALLOWED.test(file.type) || (!file.type && EXT.test(name)))) {
        return showProblem("That file type can't be used" + (file.type ? ' (' + file.type + ')' : '') + '. Supported: JPG, PNG, WebP, GIF or BMP.', ['If it is a document or screenshot, take a photo of the affected area instead.']);
      }
      if (file.size === 0) return showProblem('That file is empty (0 bytes).', ['Choose the photo again, or take a new one.']);
      if (file.size > MAX) return showProblem('That photo is ' + mb(file.size) + ', and the limit is 16 MB.', ['Use the Take a photo button, which makes a smaller file.', 'Or choose a lower-resolution copy.']);
      decode(file).then(function (r) {
        var w = r.im.naturalWidth, hgt = r.im.naturalHeight;
        if (Math.min(w, hgt) < MIN_SIDE) {
          URL.revokeObjectURL(r.url);
          return showProblem('The photo is only ' + w + '×' + hgt + ' pixels, which is too small to see any detail.', ['Take it at full size (not a thumbnail).', 'Move closer so the affected area fills the frame.']);
        }
        resetPreview();
        chosen = file; objectUrl = r.url; previewImg.src = r.url;
        doc.getElementById('previewName').textContent = name;
        doc.getElementById('previewSize').textContent = w + '×' + hgt + ' px · ' + mb(file.size);
        var checks = quickCheck(r.im); clearNode(prechecks);
        checks.forEach(function (c) { prechecks.appendChild(h('li', { class: 'pc-' + c.s }, h('span', { 'aria-hidden': 'true', text: c.s === 'ok' ? '✓ ' : '! ' }), doc.createTextNode(c.t))); });
        weak = checks.some(function (c) { return c.s === 'bad'; });
        precheckNote.hidden = false; preview.classList.add('on'); uploader.hidden = true; analyzeBtn.disabled = false;
        if (weak) {
          analyzeBtn.lastChild.textContent = 'Analyze anyway';
          setStatus('This photo will probably be too poor to assess. A better photo gives a more useful answer.');
        }
      }, function () {
        showProblem('This image could not be opened. It may be damaged or in an unsupported format.', ['Choose the photo again, or take a new photo.']);
      });
    }

    /* ---- sources: file dialog, native camera, drop, paste ---- */
    dz.addEventListener('click', function () { input.click(); });
    dz.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
    chooseBtn.addEventListener('click', function () { input.click(); });
    input.addEventListener('change', function () { if (input.files.length) choose(input.files[0]); });
    captureInput.addEventListener('change', function () { if (captureInput.files.length) choose(captureInput.files[0]); });
    function hasFiles(e) { return e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], 'Files') >= 0; }
    ['dragenter', 'dragover'].forEach(function (ev) { doc.addEventListener(ev, function (e) { if (hasFiles(e)) { e.preventDefault(); if (dz.contains(e.target) || e.target === dz) dz.classList.add('dragover'); } }); });
    ['dragleave', 'drop'].forEach(function (ev) { doc.addEventListener(ev, function () { dz.classList.remove('dragover'); }); });
    doc.addEventListener('drop', function (e) {
      if (!hasFiles(e)) return;
      e.preventDefault();                                      // never let the browser navigate to a dropped image
      var files = e.dataTransfer.files;
      if (!files.length) { setStatus("That isn't a photo file."); return; }
      if (files.length > 1) setStatus('Please add one photo at a time. Using the first one.');
      choose(files[0]);
    });
    doc.addEventListener('paste', function (e) {
      var t = e.target && e.target.tagName;
      if (t === 'TEXTAREA' || t === 'INPUT') return;
      var items = (e.clipboardData && e.clipboardData.items) || [];
      for (var i = 0; i < items.length; i++) if (items[i].kind === 'file' && /^image\//.test(items[i].type)) { choose(items[i].getAsFile()); e.preventDefault(); return; }
    });
    doc.getElementById('removeImage').addEventListener('click', function () { if (xhr) xhr.abort(); resetPreview(); setStatus(''); resultBox.classList.remove('on'); });

    /* ---- camera ---- */
    var camDialog = doc.getElementById('cameraDialog'), camVideo = doc.getElementById('camVideo'), camMsg = doc.getElementById('camMsg');
    var camShot = doc.getElementById('camShot'), camSwitch = doc.getElementById('camSwitch'), camCanvas = doc.getElementById('camCanvas');
    var stream = null, facing = 'environment';
    function coarse() { return !!(window.matchMedia && window.matchMedia('(pointer: coarse)').matches); }
    function canStream() { return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && camDialog && camDialog.showModal); }
    function stopStream() { if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; } camVideo.srcObject = null; }
    function camError(err) {
      var msg = {
        NotAllowedError: 'Camera access was blocked. Allow the camera for this site in your browser settings, or choose a photo instead.',
        PermissionDeniedError: 'Camera access was blocked. Allow the camera for this site in your browser settings, or choose a photo instead.',
        NotFoundError: 'No camera was found on this device. Choose a photo instead.',
        NotReadableError: 'The camera is being used by another app. Close it and try again.',
        OverconstrainedError: 'This camera does not support the requested settings. Try Switch camera.',
        SecurityError: 'The camera only works on a secure (https) page.'
      }[err && err.name] || 'The camera could not be started. Choose a photo instead.';
      camMsg.textContent = msg; camMsg.hidden = false; camShot.disabled = true;
    }
    function startStream() {
      stopStream(); camMsg.hidden = true; camShot.disabled = true;
      navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false }).then(function (s) {
        stream = s; camVideo.srcObject = s; camShot.disabled = false;
        navigator.mediaDevices.enumerateDevices().then(function (d) { camSwitch.hidden = d.filter(function (x) { return x.kind === 'videoinput'; }).length < 2; }).catch(function () {});
      }).catch(camError);
    }
    function openCameraOrNative() { if (coarse() || !canStream()) captureInput.click(); else { camDialog.showModal(); startStream(); } }
    cameraBtn.addEventListener('click', openCameraOrNative);
    camSwitch.addEventListener('click', function () { facing = facing === 'user' ? 'environment' : 'user'; startStream(); });
    doc.getElementById('camClose').addEventListener('click', function () { camDialog.close(); });
    camDialog.addEventListener('close', stopStream);
    camShot.addEventListener('click', function () {
      if (!camVideo.videoWidth) return;
      camCanvas.width = camVideo.videoWidth; camCanvas.height = camVideo.videoHeight;
      camCanvas.getContext('2d').drawImage(camVideo, 0, 0);
      camCanvas.toBlob(function (blob) {
        if (!blob) { camError({ name: 'x' }); return; }
        var file = new File([blob], 'camera-' + Date.now() + '.jpg', { type: 'image/jpeg' });
        camDialog.close(); choose(file);
      }, 'image/jpeg', 0.92);
    });

    /* ---- upload with real progress, cancel, and retry ---- */
    var cancelBtn = doc.getElementById('cancelUpload');
    cancelBtn.addEventListener('click', function () { if (xhr) xhr.abort(); });
    function upload() {
      if (!chosen || xhr) return;
      var fd = new FormData(); fd.append('file', chosen, chosen.name || 'photo.jpg');
      xhr = new XMLHttpRequest(); xhr.open('POST', '/upload'); xhr.setRequestHeader('X-CSRFToken', CSRF); xhr.timeout = 120000;
      setStatus(''); resultBox.classList.remove('on');
      analyzeBtn.disabled = true; progressBox.hidden = false; progressBox.classList.remove('indeterminate'); bar.style.width = '0%'; stageEl.textContent = 'Uploading… 0%';
      xhr.upload.onprogress = function (e) {
        if (!e.lengthComputable) return;
        var pct = Math.round(e.loaded / e.total * 100); bar.style.width = pct + '%'; stageEl.textContent = 'Uploading… ' + pct + '% (' + mb(e.loaded) + ' of ' + mb(e.total) + ')';
      };
      xhr.upload.onload = function () { bar.style.width = '100%'; progressBox.classList.add('indeterminate'); stageEl.textContent = 'Uploaded. Checking photo quality and analysing…'; };
      xhr.onload = function () {
        var data; try { data = JSON.parse(xhr.responseText); } catch (e) { data = { error: 'The server sent an unexpected response.' }; }
        if (xhr.status === 401) { window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname); return; }
        if (xhr.status >= 200 && xhr.status < 300 && data.analysis) {
          renderResult(resultBox, data.analysis, 'image', data.analysis_id); countCheck('image', data.analysis);
          var q = data.analysis.quality_check, poor = data.analysis.assessment_status === 'insufficient' || data.analysis.assessment_status === 'invalid' || (q && q.rating === 'poor');
          if (poor) {
            var row = h('div', { class: 'actions' });
            var again = h('button', { type: 'button', class: 'btn' }, 'Take another photo'); again.addEventListener('click', function () { resetPreview(); openCameraOrNative(); });
            var pick = h('button', { type: 'button', class: 'btn btn-ghost' }, 'Choose another photo'); pick.addEventListener('click', function () { resetPreview(); input.click(); });
            row.appendChild(again); row.appendChild(pick); resultBox.insertBefore(row, resultBox.children[1] || null);
          }
        } else {
          showProblem(data.error || 'The photo could not be analysed.', data.hints || [], xhr.status >= 500);
        }
      };
      xhr.onerror = function () { showProblem('Network problem. The photo was not sent.', ['Check your connection and try again.'], true); };
      xhr.ontimeout = function () { showProblem('This is taking too long.', ['Check your connection and try again.'], true); };
      xhr.onabort = function () { setStatus('Upload cancelled. You can try again.'); };
      xhr.onloadend = function () { xhr = null; progressBox.hidden = true; progressBox.classList.remove('indeterminate'); analyzeBtn.disabled = !chosen; };
      xhr.send(fd);
    }
    analyzeBtn.addEventListener('click', upload);
  }

  /* ----- symptom tool: adaptive interview ----- */
  var symInput = doc.getElementById('symptomsInput');
  if (symInput) {
    var symIntro = doc.getElementById('symIntro');
    var symBtn = doc.getElementById('checkSymptomsBtn');
    var symLoading = doc.getElementById('symptomLoading');
    var symResult = doc.getElementById('symptomResult');
    var count = doc.getElementById('symCount');
    var ivBox = doc.getElementById('symInterview');
    var ivQ = doc.getElementById('ivQuestion'), ivWhy = doc.getElementById('ivWhy'), ivOpts = doc.getElementById('ivOptions');
    var ivNext = doc.getElementById('ivNext'), ivSkip = doc.getElementById('ivSkip'), ivFinish = doc.getElementById('ivFinish');
    var ivCancel = doc.getElementById('ivCancel'), ivErr = doc.getElementById('ivError');
    var ivText = doc.getElementById('ivProgressText'), ivBar = doc.getElementById('ivBar');
    var iv = { session: null, q: null, value: null };

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

    function busy(on) {
      symLoading.classList.toggle('on', on);
      [ivNext, ivSkip, ivFinish, ivCancel, symBtn].forEach(function (b) { if (b) b.disabled = on; });
    }
    function showIntro() { ivBox.hidden = true; symIntro.hidden = false; iv.session = null; }
    function showError(msg) {
      if (ivBox.hidden) renderError(symResult, msg);
      else { ivErr.textContent = msg; ivErr.hidden = false; }
    }

    function renderQuestion(q, progress) {
      iv.q = q; iv.value = null; ivErr.hidden = true;
      symIntro.hidden = true; ivBox.hidden = false; symResult.classList.remove('on');
      var n = (progress.answered || 0) + 1;
      ivText.textContent = 'Question ' + n + ' of about ' + Math.max(n, progress.estimated_total || n);
      ivBar.style.width = Math.min(100, Math.round(((progress.answered || 0) / Math.max(1, progress.estimated_total || 1)) * 100)) + '%';
      ivQ.textContent = q.text;
      ivWhy.textContent = q.why || ''; ivWhy.hidden = !q.why;
      ivOpts.replaceChildren();
      ivOpts.className = 'iv-options iv-' + q.type;
      ivSkip.textContent = q.optional ? 'Nothing to add' : 'Skip';
      ivNext.disabled = true;

      if (q.type === 'single') {
        ivOpts.setAttribute('role', 'radiogroup');
        q.options.forEach(function (o) {
          var b = h('button', { type: 'button', class: 'opt', role: 'radio', 'aria-checked': 'false', text: o.label });
          b.addEventListener('click', function () {
            ivOpts.querySelectorAll('.opt').forEach(function (x) { x.classList.remove('on'); x.setAttribute('aria-checked', 'false'); });
            b.classList.add('on'); b.setAttribute('aria-checked', 'true'); iv.value = o.value; ivNext.disabled = false;
          });
          ivOpts.appendChild(b);
        });
      } else if (q.type === 'multi') {
        ivOpts.setAttribute('role', 'group');
        var chosen = {};
        q.options.forEach(function (o) {
          var b = h('button', { type: 'button', class: 'opt', role: 'checkbox', 'aria-checked': 'false', text: o.label });
          b.addEventListener('click', function () {
            var on = !chosen[o.value];
            if (on && o.exclusive) { chosen = {}; ivOpts.querySelectorAll('.opt').forEach(function (x) { x.classList.remove('on'); x.setAttribute('aria-checked', 'false'); }); }
            if (on && !o.exclusive) {
              q.options.forEach(function (x, i) { if (x.exclusive && chosen[x.value]) { delete chosen[x.value]; var xb = ivOpts.children[i]; xb.classList.remove('on'); xb.setAttribute('aria-checked', 'false'); } });
            }
            if (on) chosen[o.value] = true; else delete chosen[o.value];
            b.classList.toggle('on', on); b.setAttribute('aria-checked', on ? 'true' : 'false');
            iv.value = Object.keys(chosen); ivNext.disabled = iv.value.length === 0;
          });
          ivOpts.appendChild(b);
        });
      } else if (q.type === 'scale') {
        ivOpts.setAttribute('role', 'radiogroup');
        for (var i = q.min; i <= q.max; i++) {
          (function (n2) {
            var b = h('button', { type: 'button', class: 'opt scale', role: 'radio', 'aria-checked': 'false', 'aria-label': n2 + ' out of ' + q.max, text: String(n2) });
            b.addEventListener('click', function () {
              ivOpts.querySelectorAll('.opt').forEach(function (x) { x.classList.remove('on'); x.setAttribute('aria-checked', 'false'); });
              b.classList.add('on'); b.setAttribute('aria-checked', 'true'); iv.value = n2; ivNext.disabled = false;
            });
            ivOpts.appendChild(b);
          })(i);
        }
        ivOpts.appendChild(h('div', { class: 'scale-legend' }, h('span', { text: 'Barely noticeable' }), h('span', { text: 'Worst imaginable' })));
      } else {
        var ta = h('textarea', { rows: '3', maxlength: '300', placeholder: 'Type here (optional)', 'aria-label': q.text });
        ta.addEventListener('input', function () { iv.value = ta.value; });
        iv.value = '';
        ivOpts.appendChild(ta);
        ivNext.disabled = false;
      }
      var first = ivOpts.querySelector('button, textarea'); if (first) first.focus({ preventScroll: true });
    }

    function finishUi(d) {
      ivBox.hidden = true; symIntro.hidden = true; iv.session = null;
      renderResult(symResult, d.analysis, 'symptom', d.analysis_id);
      countCheck('symptom', d.analysis);
      var again = h('button', { type: 'button', class: 'btn btn-ghost btn-sm' }, 'Start a new check');
      again.addEventListener('click', function () { symInput.value = ''; updateCount(); symResult.classList.remove('on'); symIntro.hidden = false; symInput.focus(); });
      symResult.appendChild(h('p', { class: 'result-actions' }, again));
    }

    function handle(d) {
      if (d.error) { showError(d.error); return; }
      if (d.status === 'question') { iv.session = d.session_id; renderQuestion(d.question, d.progress); return; }
      finishUi(d);   // 'complete' or 'emergency'
    }

    function call(url, body) {
      busy(true);
      return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF }, body: JSON.stringify(body) })
        .then(parseJsonResponse).then(handle)
        .catch(function (err) { if (err.message !== 'auth') showError('Network problem. Check your connection and try again.'); })
        .then(function () { busy(false); if (iv.q && !ivBox.hidden) ivNext.disabled = (iv.value === null || (Array.isArray(iv.value) && !iv.value.length)) && iv.q.type !== 'text'; });
    }

    symBtn.addEventListener('click', function () {
      var text = symInput.value.trim();
      if (text.length < 5) { renderError(symResult, 'Please describe your symptoms in a little more detail.'); return; }
      symResult.classList.remove('on');
      call('/api/interview/start', { symptoms: text });
    });
    ivNext.addEventListener('click', function () {
      if (!iv.session || !iv.q) return;
      call('/api/interview/answer', { session_id: iv.session, question_id: iv.q.id, answer: iv.value });
    });
    ivSkip.addEventListener('click', function () {
      if (!iv.session || !iv.q) return;
      call('/api/interview/answer', { session_id: iv.session, question_id: iv.q.id, skip: true });
    });
    ivFinish.addEventListener('click', function () { if (iv.session) call('/api/interview/finish', { session_id: iv.session }); });
    ivCancel.addEventListener('click', showIntro);
  }

  /* ----- region setting (account page) ----- */
  var regionSel = doc.querySelector('[data-region-select]');
  if (regionSel) {
    regionSel.addEventListener('change', function () {
      var msg = doc.getElementById('regionMsg');
      fetch('/api/region', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF }, body: JSON.stringify({ region: regionSel.value }) })
        .then(function (r) { return r.json(); })
        .then(function (d) {
          msg.className = 'match-msg ' + (d.error ? 'bad' : 'ok');
          msg.textContent = d.error ? d.error : 'Saved. Emergency number is now ' + d.emergency_number + '.';
          if (!d.error) doc.body.setAttribute('data-emergency', d.emergency_number);
        })
        .catch(function () { msg.className = 'match-msg bad'; msg.textContent = 'Could not save. Try again.'; });
    });
  }
})();

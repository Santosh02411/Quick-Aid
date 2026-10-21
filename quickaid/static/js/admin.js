/* Quick Aid - internal AI evaluation dashboard. DOM + SVG only; every server value goes in via textContent. */
(function () {
  'use strict';
  var app = document.getElementById('adminApp');
  if (!app) return;
  var doc = document, NS = 'http://www.w3.org/2000/svg';
  var CSRF = (doc.querySelector('meta[name="csrf-token"]') || {}).content || '';
  var $ = function (id) { return doc.getElementById(id); };
  var state = { runs: [], datasets: [], config: null, cmpA: null, cmpB: null };

  /* ---------- helpers ---------- */
  function h(tag, props) {
    var el = doc.createElement(tag);
    Object.keys(props || {}).forEach(function (k) {
      if (k === 'class') el.className = props[k]; else if (k === 'text') el.textContent = props[k]; else el.setAttribute(k, props[k]);
    });
    var add = function (c) {
      if (c == null || c === false) return;
      if (Array.isArray(c)) { c.forEach(add); return; }          // arrays of children (table cells, list items)
      el.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c);
    };
    for (var i = 2; i < arguments.length; i++) add(arguments[i]);
    return el;
  }
  function svg(tag, attrs) { var e = doc.createElementNS(NS, tag); Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); }); return e; }
  function api(url, opts) {
    opts = opts || {}; opts.headers = Object.assign({ 'X-CSRFToken': CSRF }, opts.headers || {});
    return fetch(url, opts).then(function (r) { return r.json().then(function (d) { d._status = r.status; return d; }); });
  }
  var pct = function (v, d) { return v == null ? 'n/a' : (v * 100).toFixed(d == null ? 1 : d) + '%'; };
  var ms = function (v) { return v == null ? 'n/a' : (v >= 1000 ? (v / 1000).toFixed(2) + ' s' : v.toFixed(0) + ' ms'); };
  var usd = function (v) { return v == null ? 'n/a' : '$' + (v < 0.01 ? v.toFixed(5) : v.toFixed(4)); };
  var when = function (iso) { return (iso || '').replace('T', ' ').slice(0, 16); };
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  /* ---------- KPI cards ---------- */
  function kpi(label, value, sub, tone) {
    return h('div', { class: 'kpi ' + (tone || (value === 'n/a' ? 'na' : '')) }, h('div', { class: 'kl', text: label }), h('div', { class: 'kv', text: value }), sub ? h('div', { class: 'ks', text: sub }) : null);
  }
  // tone from a threshold: higher-is-better metrics are bad below `min`; lower-is-better are bad above `max`
  function tone(v, opts) {
    if (v == null) return 'na';
    if (opts.min != null) return v >= opts.min ? 'ok' : 'bad';
    if (opts.max != null) return v <= opts.max ? 'ok' : (v <= opts.max * 2 ? 'warn' : 'bad');
    return '';
  }

  function renderConfig() {
    var c = state.config, box = $('cfg'); clear(box);
    var chip = function (k, v) { return h('span', { class: 'chipx' }, h('b', { text: k }), doc.createTextNode(v)); };
    box.appendChild(chip('Model', c.model + (c.ai_connected ? '' : ' (not connected)')));
    box.appendChild(chip('Prompts', c.prompts.fingerprint + (c.prompts.label ? ' · ' + c.prompts.label : '')));
    box.appendChild(chip('Safety engine', 'v' + c.engine_version));
    box.appendChild(chip('Cost tracking', c.price_configured ? 'prices set' : 'prices not set (tokens only)'));
    box.appendChild(chip('Telemetry kept', c.retention_days + ' days'));
  }

  function renderDatasetProblems(problems) {
    var box = $('dsProblems'); clear(box);
    if (problems && problems.length) box.appendChild(h('div', { class: 'alert alert-error', role: 'alert' }, h('strong', { text: 'Dataset integrity: ' }), doc.createTextNode(problems.join(' · '))));
  }

  function runLabel(r) { return when(r.created) + ' · ' + r.mode + (r.label ? ' · ' + r.label : ''); }

  function renderKpis(run) {
    var box = $('kpis'), banner = $('runBanner'); clear(box); clear(banner);
    if (!run) { box.appendChild(h('p', { class: 'empty', text: 'No evaluation runs yet. Start one below.' })); return; }
    var m = run.metrics || {}, roles = {};
    state.datasets.forEach(function (d) { roles[d.name] = d.role; });
    var info = [run.model.name, 'prompts ' + run.prompts.fingerprint, 'engine v' + run.engine_version, (run.sets || []).map(function (s) { return s + ' v' + ((run.datasets[s] || {}).version || '?'); }).join(', '), run.cases_evaluated + ' cases'];
    banner.appendChild(h('p', { class: 'hint' }, doc.createTextNode(info.join('  ·  '))));
    if (run.model.synthetic) banner.appendChild(h('div', { class: 'alert alert-warn', text: 'Synthetic mock model: these numbers test the plumbing and the safety rules, not a real AI model.' }));
    if (run.mode === 'offline') banner.appendChild(h('div', { class: 'alert alert-warn', text: 'Rules only: no AI model was involved, so validity, failure, latency and cost are not applicable.' }));
    if ((run.sets || []).some(function (s) { return /dev/.test(roles[s] || ''); })) banner.appendChild(h('div', { class: 'alert alert-warn', text: 'Includes the development set: the rules and prompts were tuned while it was written, so these results are optimistic. Judge generalisation on held-out sets.' }));
    if (run.threshold_failures && run.threshold_failures.length) banner.appendChild(h('div', { class: 'alert alert-error' }, h('strong', { text: 'Safety thresholds missed' + (run.gated ? '' : ' (informational for this model)') + ': ' }), doc.createTextNode(run.threshold_failures.join(' · '))));
    if (run.notes) banner.appendChild(h('p', { class: 'hint', text: run.notes }));
    var t = run.telemetry || {};
    var cards = [
      kpi('Emergency recall', pct(m.emergency_recall), 'must be 100%', tone(m.emergency_recall, { min: 1 })),
      kpi('False-negative rate', pct(m.false_negative_rate), 'missed emergencies', tone(m.false_negative_rate == null ? null : 1 - m.false_negative_rate, { min: 1 })),
      kpi('False-positive rate', pct(m.false_positive_rate), 'false alarms (target ≤ 10%)', tone(m.false_positive_rate, { max: 0.10 })),
      kpi('Under-triage', pct(m.under_triage), 'less urgent than acceptable (≤ 5%)', tone(m.under_triage, { max: 0.05 })),
      kpi('Structured-output validity', pct(m.structured_output_validity), (t.invalid_outputs || 0) + ' invalid of ' + (t.calls || 0) + ' calls', m.structured_output_validity == null ? 'na' : tone(m.structured_output_validity, { min: 0.98 })),
      kpi('API failure rate', pct(m.api_failure_rate), (t.calls || 0) + ' calls', m.api_failure_rate == null ? 'na' : tone(m.api_failure_rate, { max: 0.02 })),
      kpi('Latency p50 / p95', ms(m.latency_p50_ms) + ' / ' + ms(m.latency_p95_ms), run.mode === 'replay' ? 'recorded latency' : (run.model.synthetic ? 'mock: not meaningful' : ''), ''),
      kpi('Cost per analysis', usd(m.cost_per_analysis_usd), t.cost_configured === false ? 'prices not configured' : (t.input_tokens || 0) + ' in / ' + (t.output_tokens || 0) + ' out tokens', '')
    ];
    cards.forEach(function (c) { box.appendChild(c); });
  }

  /* ---------- line charts (inline SVG) ---------- */
  var COLORS = ['#18785f', '#cf3427', '#9a5b00', '#4a6fa5'];
  function lineChart(title, series, runs, fmt) {
    var W = 360, H = 170, P = { l: 38, r: 10, t: 10, b: 22 };
    var box = h('div', { class: 'chart' }, h('h3', { text: title }));
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': title });
    var n = runs.length;
    if (n < 1) { box.appendChild(h('p', { class: 'hint', text: 'No data yet.' })); return box; }
    var vals = []; series.forEach(function (se) { runs.forEach(function (r) { var v = r.metrics[se.key]; if (v != null) vals.push(v); }); });
    if (!vals.length) { box.appendChild(h('p', { class: 'hint', text: 'Not measured in these runs.' })); return box; }
    var lo = 0, hi = Math.max(1, Math.max.apply(null, vals));
    if (series[0].key === 'latency_p50_ms') hi = Math.max.apply(null, vals) * 1.1 || 1;
    var x = function (i) { return P.l + (n === 1 ? (W - P.l - P.r) / 2 : i * (W - P.l - P.r) / (n - 1)); };
    var y = function (v) { return P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b); };
    [0, 0.5, 1].forEach(function (f) {
      var val = lo + (hi - lo) * f, yy = y(val);
      s.appendChild(svg('line', { x1: P.l, x2: W - P.r, y1: yy, y2: yy, stroke: 'currentColor', 'stroke-opacity': '.12' }));
      var tx = svg('text', { x: P.l - 5, y: yy + 3, 'text-anchor': 'end', 'font-size': '9', fill: 'currentColor', 'fill-opacity': '.6' }); tx.textContent = fmt(val); s.appendChild(tx);
    });
    series.forEach(function (se, si) {
      var pts = [], color = COLORS[si % COLORS.length];
      runs.forEach(function (r, i) { var v = r.metrics[se.key]; if (v != null) pts.push([x(i), y(v), v, r]); });
      if (pts.length > 1) s.appendChild(svg('polyline', { points: pts.map(function (p) { return p[0] + ',' + p[1]; }).join(' '), fill: 'none', stroke: color, 'stroke-width': '2' }));
      pts.forEach(function (p) { var c = svg('circle', { cx: p[0], cy: p[1], r: '3.5', fill: color }); var tt = svg('title'); tt.textContent = se.label + ': ' + fmt(p[2]) + ' · ' + runLabel(p[3]); c.appendChild(tt); s.appendChild(c); });
    });
    var first = svg('text', { x: P.l, y: H - 6, 'font-size': '9', fill: 'currentColor', 'fill-opacity': '.6' }); first.textContent = when(runs[0].created).slice(5, 10);
    var last = svg('text', { x: W - P.r, y: H - 6, 'text-anchor': 'end', 'font-size': '9', fill: 'currentColor', 'fill-opacity': '.6' }); last.textContent = when(runs[n - 1].created).slice(5, 10);
    s.appendChild(first); if (n > 1) s.appendChild(last);
    box.appendChild(s);
    var lg = h('div', { class: 'legend' });
    series.forEach(function (se, si) { lg.appendChild(h('span', {}, (function () { var sw = h('span', { class: 'sw' }); sw.style.background = COLORS[si % COLORS.length]; return sw; })(), doc.createTextNode(se.label))); });
    box.appendChild(lg);
    return box;
  }

  function renderTrends() {
    var mode = $('trendMode').value, box = $('charts'); clear(box);
    var runs = state.runs.filter(function (r) { return !mode || r.mode === mode; }).slice().reverse();
    box.appendChild(lineChart('Emergency recall and false-negative rate', [{ key: 'emergency_recall', label: 'recall' }, { key: 'false_negative_rate', label: 'false-negative rate' }], runs, pct));
    box.appendChild(lineChart('False alarms and under-triage', [{ key: 'false_positive_rate', label: 'false-positive rate' }, { key: 'under_triage', label: 'under-triage' }], runs, pct));
    box.appendChild(lineChart('Output validity and API failures', [{ key: 'structured_output_validity', label: 'valid output' }, { key: 'api_failure_rate', label: 'API failure rate' }], runs, pct));
    box.appendChild(lineChart('Latency (p50)', [{ key: 'latency_p50_ms', label: 'p50 ms' }], runs, function (v) { return v.toFixed(0); }));
  }

  /* ---------- runs table + compare ---------- */
  function renderRuns() {
    var t = $('runsTable'); clear(t);
    var head = ['A', 'B', 'When', 'Mode / model', 'Prompts', 'Datasets', 'Recall', 'FN', 'FP', 'Valid', 'API fail', 'p50', 'Cost/analysis', 'Note'];
    t.appendChild(h('thead', {}, h('tr', {}, head.map(function (c) { return h('th', { text: c }); }))));
    var tb = h('tbody');
    state.runs.forEach(function (r) {
      var m = r.metrics, mk = function (name, val) { var i = h('input', { type: 'radio', name: name, value: r.run_id, 'aria-label': name === 'cmpA' ? 'Baseline A' : 'Candidate B' }); i.checked = (name === 'cmpA' ? state.cmpA : state.cmpB) === r.run_id; i.addEventListener('change', function () { state[name] = r.run_id; }); return i; };
      var modeCell = h('td', {}, h('span', { class: 'runtag' + (r.synthetic_model ? ' synthetic' : ''), text: r.mode }), doc.createTextNode(' ' + (r.model || '')));
      tb.appendChild(h('tr', {}, h('td', {}, mk('cmpA')), h('td', {}, mk('cmpB')), h('td', { text: when(r.created) }), modeCell,
        h('td', { class: 'mono', text: r.prompt_fingerprint || '' }), h('td', { text: Object.keys(r.datasets || {}).filter(function (k) { return (r.sets || []).indexOf(k) >= 0; }).map(function (k) { return k + ' v' + r.datasets[k]; }).join(', ') }),
        h('td', { class: 'num', text: pct(m.emergency_recall) }), h('td', { class: 'num', text: pct(m.false_negative_rate) }), h('td', { class: 'num', text: pct(m.false_positive_rate) }),
        h('td', { class: 'num', text: pct(m.structured_output_validity) }), h('td', { class: 'num', text: pct(m.api_failure_rate) }), h('td', { class: 'num', text: ms(m.latency_p50_ms) }),
        h('td', { class: 'num', text: usd(m.cost_per_analysis_usd) }), h('td', { text: r.label || '' })));
    });
    t.appendChild(tb);
    if (!state.runs.length) t.appendChild(h('caption', { class: 'empty', text: 'No runs yet.' }));
  }

  function renderCompare(c) {
    var out = $('compareOut'); clear(out);
    if (c.error) { out.appendChild(h('div', { class: 'alert alert-error', text: c.error })); return; }
    c.warnings.forEach(function (w) { out.appendChild(h('div', { class: 'alert alert-warn', text: w })); });
    var t = h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, ['Metric', 'A (baseline)', 'B (candidate)', 'Change', ''].map(function (x) { return h('th', { text: x }); }))));
    var tb = h('tbody');
    c.metrics.forEach(function (r) {
      var isPct = !/latency|cost/.test(r.metric), f = /cost/.test(r.metric) ? usd : (isPct ? pct : ms);
      var d = r.delta == null ? 'n/a' : (r.delta > 0 ? '+' : '') + (isPct ? (r.delta * 100).toFixed(1) + ' pts' : (/cost/.test(r.metric) ? usd(r.delta) : r.delta.toFixed(0) + ' ms'));
      tb.appendChild(h('tr', {}, h('td', { text: r.label }), h('td', { class: 'num', text: f(r.a) }), h('td', { class: 'num', text: f(r.b) }), h('td', { class: 'num', text: d }), h('td', { class: 'verdict-' + r.verdict, text: r.verdict })));
    });
    t.appendChild(tb); out.appendChild(h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Scrollable table' }, t));
    var diffs = [];
    Object.keys(c.prompt_diff).forEach(function (k) { diffs.push('prompt "' + k + '": ' + (c.prompt_diff[k].a || '-') + ' → ' + (c.prompt_diff[k].b || '-')); });
    Object.keys(c.dataset_diff).forEach(function (k) { diffs.push('dataset ' + k + ': v' + (c.dataset_diff[k].a || '-') + ' → v' + (c.dataset_diff[k].b || '-')); });
    if (c.model.a !== c.model.b) diffs.push('model: ' + c.model.a + ' → ' + c.model.b);
    if (c.engine_version.a !== c.engine_version.b) diffs.push('safety engine: v' + c.engine_version.a + ' → v' + c.engine_version.b);
    out.appendChild(h('h3', { text: 'What changed between the runs' }));
    out.appendChild(diffs.length ? h('ul', { class: 'cmp-list' }, diffs.map(function (d) { return h('li', { text: d }); })) : h('p', { class: 'hint', text: 'Nothing recorded as different (same model, prompts, dataset and engine).' }));
    var caseList = function (title, rows) {
      out.appendChild(h('h3', { text: title + ' (' + rows.length + ')' }));
      if (rows.length) out.appendChild(h('ul', { class: 'cmp-list' }, rows.slice(0, 25).map(function (r) {
        return h('li', { text: r.id + ' "' + (r.text || '') + '": ' + (r.a.level || 'no answer') + (r.a.alert ? ' +alert' : '') + ' → ' + (r.b.level || 'no answer') + (r.b.alert ? ' +alert' : '') + (r.b.problems && r.b.problems.length ? ' [' + r.b.problems.join(', ') + ']' : '') });
      })));
    };
    caseList('Regressions: cases that were right and are now wrong', c.regressions);
    caseList('Improvements', c.improvements);
    caseList('Changed answer, same verdict', c.changed);
  }

  /* ---------- live telemetry ---------- */
  function renderTelemetry(t) {
    var box = $('telKpis'), ch = $('telCharts'), tb = $('telTables'); clear(box); clear(ch); clear(tb);
    if (!t.calls && !t.analyses) { box.appendChild(h('p', { class: 'empty', text: 'No AI activity recorded in this window yet.' })); return; }
    [kpi('Analyses', String(t.analyses), (t.completed_analyses || 0) + ' completed · ' + (t.ai_attempted || 0) + ' used the AI'),
     kpi('API failure rate', pct(t.api_failure_rate), (t.calls || 0) + ' model calls', t.api_failure_rate == null ? 'na' : tone(t.api_failure_rate, { max: 0.02 })),
     kpi('Structured-output validity', pct(t.structured_output_validity), (t.invalid_outputs || 0) + ' invalid', t.structured_output_validity == null ? 'na' : tone(t.structured_output_validity, { min: 0.98 })),
     kpi('Latency p50 / p95', ms((t.latency_ms || {}).p50) + ' / ' + ms((t.latency_ms || {}).p95), 'end-to-end p95 ' + ms((t.end_to_end_latency_ms || {}).p95)),
     kpi('Fell back to basic mode', pct(t.fallback_rate), 'of analyses that tried the AI', t.fallback_rate == null ? 'na' : tone(t.fallback_rate, { max: 0.05 })),
     kpi('Safety engine raised urgency', pct(t.safety_intervention_rate), 'of analyses'),
     kpi('Tokens in / out', (t.input_tokens || 0) + ' / ' + (t.output_tokens || 0), ''),
     kpi('Cost per analysis', usd(t.cost_per_analysis_usd), t.cost_configured ? 'total ' + usd(t.cost_usd) : 'set GEMINI_PRICE_INPUT_PER_M / OUTPUT_PER_M')].forEach(function (c) { box.appendChild(c); });
    // daily bars: calls and failures
    var days = t.daily || [];
    if (days.length) {
      var W = 720, H = 170, bw = Math.max(8, Math.min(46, (W - 60) / days.length - 8)), max = Math.max.apply(null, days.map(function (d) { return d.calls; })) || 1;
      var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Model calls per day' });
      days.forEach(function (d, i) {
        var x = 44 + i * ((W - 56) / days.length), hh = (H - 34) * d.calls / max, fh = (H - 34) * (d.failures + d.invalid) / max;
        var r1 = svg('rect', { x: x, y: H - 20 - hh, width: bw, height: hh, fill: COLORS[0], 'fill-opacity': '.75' }); var t1 = svg('title'); t1.textContent = d.day + ': ' + d.calls + ' calls, ' + d.failures + ' failed, ' + d.invalid + ' invalid'; r1.appendChild(t1);
        s.appendChild(r1); if (fh > 0) s.appendChild(svg('rect', { x: x, y: H - 20 - fh, width: bw, height: fh, fill: COLORS[1] }));
        var lb = svg('text', { x: x + bw / 2, y: H - 5, 'text-anchor': 'middle', 'font-size': '11', fill: 'currentColor', 'fill-opacity': '.65' }); lb.textContent = d.day.slice(5); s.appendChild(lb);
      });
      var cb = h('div', { class: 'chart' }, h('h3', { text: 'Model calls per day' }), s, h('div', { class: 'legend' }, h('span', { text: 'green = calls' }), h('span', { text: 'red = failed or invalid' })));
      ch.appendChild(cb);
    }
    var group = function (title, obj, keyLabel) {
      var keys = Object.keys(obj || {}); if (!keys.length) return;
      tb.appendChild(h('h3', { text: title }));
      var t2 = h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, [keyLabel, 'Calls', 'API fail', 'Valid', 'p50', 'p95', 'Cost'].map(function (x) { return h('th', { text: x }); }))));
      var b2 = h('tbody');
      keys.forEach(function (k) { var o = obj[k]; b2.appendChild(h('tr', {}, h('td', { class: 'mono', text: k }), h('td', { class: 'num', text: String(o.calls) }), h('td', { class: 'num', text: pct(o.api_failure_rate) }), h('td', { class: 'num', text: pct(o.structured_output_validity) }), h('td', { class: 'num', text: ms(o.latency_ms.p50) }), h('td', { class: 'num', text: ms(o.latency_ms.p95) }), h('td', { class: 'num', text: usd(o.cost_usd) }))); });
      t2.appendChild(b2); tb.appendChild(h('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Scrollable table' }, t2));
    };
    group('By model (compare model versions on live traffic)', t.by_model, 'Model');
    group('By prompt version (compare prompt changes on live traffic)', t.by_prompt_version, 'Prompt hash');
    group('By kind', t.by_kind, 'Kind');
    var errs = Object.keys(t.errors || {});
    if (errs.length) tb.appendChild(h('p', { class: 'hint', text: 'Errors: ' + errs.map(function (e) { return e + ' ×' + t.errors[e]; }).join(', ') }));
  }

  /* ---------- datasets ---------- */
  function renderDatasets() {
    var t = $('dsTable'); clear(t);
    t.appendChild(h('thead', {}, h('tr', {}, ['Dataset', 'Version', 'Role', 'Cases', 'Integrity', 'Labelled by', 'Clinician reviewed', 'Last change'].map(function (x) { return h('th', { text: x }); }))));
    var tb = h('tbody');
    state.datasets.forEach(function (d) {
      var last = (d.changelog || []).slice(-1)[0];
      tb.appendChild(h('tr', {}, h('td', { class: 'mono', text: d.name }), h('td', { text: 'v' + d.version + (d.frozen ? ' (frozen)' : '') }), h('td', { text: d.role }), h('td', { class: 'num', text: String(d.cases) }),
        h('td', { class: d.status === 'ok' ? 'verdict-better' : 'verdict-worse', text: d.status === 'ok' ? 'matches manifest' : 'CHANGED without a version bump' }), h('td', { text: d.labelled_by || '' }),
        h('td', { text: d.clinician_reviewed ? 'yes' : 'no' }), h('td', { text: last ? last.date + ': ' + last.note : '' })));
    });
    t.appendChild(tb);
    var sets = $('newSets'); clear(sets); sets.appendChild(h('legend', { text: 'Datasets' }));
    state.datasets.filter(function (d) { return /^symptom_/.test(d.name); }).forEach(function (d) {
      var cb = h('input', { type: 'checkbox', value: d.name }); cb.checked = d.name === 'symptom_cases';
      sets.appendChild(h('label', {}, cb, doc.createTextNode(d.name + ' v' + d.version + ' (' + d.cases + ' cases, ' + d.role + ')')));
    });
  }

  function renderImageReference() {
    var box = $('imgRef'); clear(box);
    box.appendChild(h('div', { class: 'alert alert-warn' }, h('strong', { text: 'Real photos need independent reference data. ' }),
      doc.createTextNode('No real medical images ship with this project. To evaluate real model output on photos, supply a small set of licensed images labelled by someone other than the model or prompt author (ideally a clinician), then run --image-reference. See evaluation/reference/README.md for the format.')));
  }

  /* ---------- loading ---------- */
  function loadKpiRun() {
    var id = $('kpiRun').value;
    if (!id) { renderKpis(null); renderImageRun(null); return; }
    api('/admin/api/runs/' + encodeURIComponent(id)).then(function (run) { if (!run.error) { renderKpis(run); renderImageRun(run); } });
  }
  function renderImageRun(run) {
    var box = $('imgRef'); renderImageReference();
    if (!run || !run.image_reference) return;
    var r = run.image_reference;
    box.appendChild(h('h3', { text: 'Real-image results in this run' }));
    var kp = h('div', { class: 'kpi-grid' });
    [kpi('Images', String(r.images), JSON.stringify(r.splits)), kpi('Under-triage', pct(r.under_triage), '', tone(r.under_triage, { max: 0.05 })), kpi('Unusable photos caught', pct(r.unusable_photo_recall), '', tone(r.unusable_photo_recall, { min: 1 })),
     kpi('Image-type accuracy', pct(r.image_type_accuracy), ''), kpi('Finding keyword recall', pct(r.finding_keyword_recall), ''), kpi('Interpretation without findings', String(r.interpretation_without_findings), 'should be 0', r.interpretation_without_findings ? 'bad' : 'ok')].forEach(function (c) { kp.appendChild(c); });
    box.appendChild(kp);
    (r.caveats || []).forEach(function (c) { box.appendChild(h('div', { class: 'alert alert-warn', text: c })); });
  }
  function loadTelemetry() { api('/admin/api/telemetry?days=' + $('telDays').value).then(renderTelemetry); }

  function refresh(selectRunId) {
    return api('/admin/api/overview').then(function (o) {
      if (o.error) { app.appendChild(h('div', { class: 'alert alert-error', text: o.error })); return; }
      state.runs = o.runs; state.datasets = o.datasets; state.config = o.config;
      renderConfig(); renderDatasetProblems(o.dataset_problems); renderDatasets();
      var sel = $('kpiRun'); clear(sel);
      state.runs.forEach(function (r) { sel.appendChild(h('option', { value: r.run_id, text: runLabel(r) })); });
      if (selectRunId) sel.value = selectRunId;
      var modes = {}; state.runs.forEach(function (r) { modes[r.mode] = 1; });
      var tm = $('trendMode'), cur = tm.value; clear(tm); tm.appendChild(h('option', { value: '', text: 'all runs' }));
      Object.keys(modes).forEach(function (m) { tm.appendChild(h('option', { value: m, text: m })); }); tm.value = cur;
      if (!state.cmpB && state.runs[0]) state.cmpB = state.runs[0].run_id;
      if (!state.cmpA && state.runs[1]) state.cmpA = state.runs[1].run_id;
      renderRuns(); renderTrends(); loadKpiRun();
    });
  }

  $('kpiRun').addEventListener('change', loadKpiRun);
  $('trendMode').addEventListener('change', renderTrends);
  $('telDays').addEventListener('change', loadTelemetry);
  $('compareBtn').addEventListener('click', function () {
    if (!state.cmpA || !state.cmpB || state.cmpA === state.cmpB) { $('compareHint').textContent = 'Pick two different runs.'; return; }
    $('compareHint').textContent = '';
    api('/admin/api/compare?a=' + encodeURIComponent(state.cmpA) + '&b=' + encodeURIComponent(state.cmpB)).then(renderCompare);
  });
  $('newRunBtn').addEventListener('click', function () {
    var sets = Array.prototype.map.call(doc.querySelectorAll('#newSets input:checked'), function (i) { return i.value; });
    var btn = $('newRunBtn'), out = $('newOut'); clear(out);
    btn.disabled = true; $('newLoading').classList.add('on');
    api('/admin/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: $('newModel').value, sets: sets, label: $('newLabel').value }) })
      .then(function (d) {
        if (d.error) { out.appendChild(h('div', { class: 'alert alert-error', text: d.error })); return; }
        out.appendChild(h('div', { class: 'alert alert-success', text: 'Saved run ' + d.run_id + '. Emergency recall ' + pct(d.metrics.emergency_recall) + (d.threshold_failures.length ? ' · thresholds missed: ' + d.threshold_failures.length : '')}));
        return refresh(d.run_id);
      })
      .catch(function () { out.appendChild(h('div', { class: 'alert alert-error', text: 'The request failed. Try again.' })); })
      .then(function () { btn.disabled = false; $('newLoading').classList.remove('on'); });
  });

  refresh().then(loadTelemetry);
})();

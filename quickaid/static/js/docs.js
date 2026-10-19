/* Quick Aid - API documentation viewer. Renders /api/openapi.json with DOM calls only (no external libraries). */
(function () {
  'use strict';
  var doc = document, root = doc.getElementById('apiDocs');
  if (!root) return;
  function h(tag, props) {
    var el = doc.createElement(tag);
    Object.keys(props || {}).forEach(function (k) { if (k === 'class') el.className = props[k]; else if (k === 'text') el.textContent = props[k]; else el.setAttribute(k, props[k]); });
    for (var i = 2; i < arguments.length; i++) { var c = arguments[i]; if (c == null) continue; if (Array.isArray(c)) c.forEach(function (x) { el.appendChild(x); }); else el.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c); }
    return el;
  }
  function resolve(spec, schema) {
    if (schema && schema.$ref) { var name = schema.$ref.split('/').pop(); return spec.components.schemas[name] || {}; }
    return schema || {};
  }
  function example(spec, schema, depth) {
    schema = resolve(spec, schema); depth = depth || 0;
    if (schema.example !== undefined) return schema.example;
    if (schema.enum) return schema.enum[0];
    if (schema.default !== undefined) return schema.default;
    var t = schema.type || (schema.anyOf ? 'anyOf' : 'object');
    if (t === 'object' && schema.properties && depth < 3) { var o = {}; Object.keys(schema.properties).forEach(function (k) { o[k] = example(spec, schema.properties[k], depth + 1); }); return o; }
    if (t === 'array') return [example(spec, schema.items || {}, depth + 1)];
    if (t === 'anyOf') return example(spec, schema.anyOf[0], depth + 1);
    return { string: 'string', integer: 1, number: 1, boolean: false }[t] !== undefined ? { string: 'string', integer: 1, number: 1, boolean: false }[t] : null;
  }
  fetch('/api/openapi.json').then(function (r) { return r.json(); }).then(function (spec) {
    doc.getElementById('apiIntro').textContent = spec.info.description;
    var byTag = {};
    Object.keys(spec.paths).sort().forEach(function (path) {
      Object.keys(spec.paths[path]).forEach(function (method) {
        var op = spec.paths[path][method];
        (op.tags || ['General']).forEach(function (t) { (byTag[t] = byTag[t] || []).push({ path: path, method: method, op: op }); });
      });
    });
    var holder = doc.getElementById('apiTags');
    Object.keys(byTag).forEach(function (tag) {
      var sec = h('section', { class: 'panel api-tag' }, h('h2', { text: tag }));
      byTag[tag].forEach(function (e) {
        var d = h('details', { class: 'api-op' });
        d.appendChild(h('summary', {}, h('span', { class: 'verb verb-' + e.method, text: e.method.toUpperCase() }), h('code', { text: e.path }), h('span', { class: 'muted', text: ' ' + e.op.summary })));
        if (e.op.description) d.appendChild(h('p', { text: e.op.description }));
        if (e.op.parameters && e.op.parameters.length) {
          d.appendChild(h('h4', { text: 'Parameters' }));
          d.appendChild(h('ul', {}, e.op.parameters.map(function (p) {
            var enums = p.schema && p.schema.enum ? ' one of: ' + p.schema.enum.join(', ') : '';
            var text = ' (' + p.in + (p.required ? ', required' : '') + ') ' + (p.description || '') + enums;
            return h('li', {}, h('code', { text: p.name }), doc.createTextNode(text));
          })));
        }
        if (e.op.requestBody) {
          var schema = e.op.requestBody.content['application/json'].schema;
          d.appendChild(h('h4', { text: 'Request body (JSON)' }));
          d.appendChild(h('pre', { text: JSON.stringify(example(spec, schema), null, 2) }));
        }
        d.appendChild(h('h4', { text: 'Responses' }));
        d.appendChild(h('ul', {}, Object.keys(e.op.responses).map(function (c) { return h('li', {}, h('code', { text: c }), doc.createTextNode(' ' + e.op.responses[c].description)); })));
        if (e.op.security) d.appendChild(h('p', { class: 'hint', text: 'Requires a signed-in session' + (e.method !== 'get' ? ' and the X-CSRFToken header.' : '.') }));
        sec.appendChild(d);
      });
      holder.appendChild(sec);
    });
  }).catch(function () { doc.getElementById('apiIntro').textContent = 'Could not load the API description.'; });
})();

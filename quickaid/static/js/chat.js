/* Quick Aid - follow-up chat page.
   All message text is inserted with textContent, never as HTML. */
(function () {
  'use strict';
  var app = document.getElementById('chatApp');
  if (!app) return;

  var CSRF = (document.querySelector('meta[name="csrf-token"]') || {}).content || '';
  var MAX = parseInt(app.getAttribute('data-max'), 10) || 500;
  var EMERGENCY = document.body.getAttribute('data-emergency') || '';

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    items: $('chatItems'), listEmpty: $('chatListEmpty'), list: $('chatList'), toggle: $('chatListToggle'),
    newBtn: $('newChatBtn'), panel: $('newChatPanel'), attach: $('attachSelect'), start: $('startChatBtn'), cancel: $('cancelNewChat'),
    ctx: $('chatContext'), empty: $('chatEmpty'), msgs: $('chatMessages'), typing: $('chatTyping'), suggest: $('chatSuggest'),
    form: $('chatForm'), input: $('chatInput'), send: $('chatSend'), count: $('chatCount'), countWrap: $('chatCountWrap')
  };
  var state = { conversations: [], current: null, busy: false };

  var SUGGESTIONS = {
    symptom: ['Is this serious?', 'What should I do now?', 'What should I avoid?', 'When should I see a doctor?', "What if it doesn't improve in 2 days?"],
    image: ['Is this serious?', 'How should I care for this?', 'What should I avoid?', 'When should I see a doctor?', 'What signs mean it is getting worse?'],
    general: ['When should I see a doctor for a fever?', 'How do I treat a minor burn?', 'What are the signs of dehydration?']
  };

  /* ---------- helpers ---------- */
  function h(tag, props) {
    var el = document.createElement(tag);
    Object.keys(props || {}).forEach(function (k) {
      if (k === 'class') el.className = props[k]; else if (k === 'text') el.textContent = props[k]; else el.setAttribute(k, props[k]);
    });
    for (var i = 2; i < arguments.length; i++) { if (arguments[i]) el.appendChild(typeof arguments[i] === 'string' ? document.createTextNode(arguments[i]) : arguments[i]); }
    return el;
  }
  function svgIcon(d) {
    var NS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(NS, 'svg'), path = document.createElementNS(NS, 'path');
    svg.setAttribute('class', 'icon'); svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('aria-hidden', 'true');
    path.setAttribute('d', d); svg.appendChild(path);
    return svg;
  }
  var PENCIL = 'M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z', DOWNLOAD = 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3';
  function trashIcon() {
    var NS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(NS, 'svg'), path = document.createElementNS(NS, 'path');
    svg.setAttribute('class', 'icon'); svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('aria-hidden', 'true');
    path.setAttribute('d', 'M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2');
    svg.appendChild(path);
    return svg;
  }
  function api(method, url, body) {
    var opts = { method: method, headers: { 'X-CSRFToken': CSRF } };
    if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    return fetch(url, opts).then(function (r) {
      if (r.status === 401) { window.location.href = '/login?next=/chat'; return Promise.reject(new Error('auth')); }
      return r.json().catch(function () { return { error: 'Unexpected server response.' }; }).then(function (d) { d._status = r.status; return d; });
    });
  }
  function fmtDate(iso) {
    try { return new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }); } catch (e) { return ''; }
  }
  function toast(message) {
    var t = h('div', { class: 'alert alert-error', role: 'alert', text: message });
    els.msgs.appendChild(t); els.msgs.hidden = false; scrollDown();
    setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, 6000);
  }
  function scrollDown() { els.msgs.scrollTop = els.msgs.scrollHeight; }

  /* ---------- conversation list ---------- */
  function renderList() {
    els.items.replaceChildren();
    els.listEmpty.hidden = state.conversations.length > 0;
    state.conversations.forEach(function (c) {
      var open = h('button', { type: 'button', class: 'chat-item' + (state.current === c.id ? ' active' : ''), 'data-id': String(c.id) },
        h('span', { class: 'chat-item-title', text: c.title || 'New chat' }),
        h('span', { class: 'chat-item-meta', text: (c.source_type ? (c.source_type === 'image' ? 'Photo check' : 'Symptom check') : 'General') + ' · ' + fmtDate(c.updated_at) }));
      var del = h('button', { type: 'button', class: 'icon-btn chat-del', 'aria-label': 'Delete chat', title: 'Delete chat', 'data-del': String(c.id) });
      del.appendChild(trashIcon());
      var ren = h('button', { type: 'button', class: 'icon-btn chat-del', 'aria-label': 'Rename chat', title: 'Rename chat', 'data-rename': String(c.id), 'data-title': c.title || '' });
      ren.appendChild(svgIcon(PENCIL));
      var exp = h('a', { class: 'icon-btn chat-del', href: '/api/chats/' + c.id + '/export?format=txt', 'aria-label': 'Download chat', title: 'Download chat', download: '' });
      exp.appendChild(svgIcon(DOWNLOAD));
      els.items.appendChild(h('li', {}, open, ren, exp, del));
    });
  }
  var searchTerm = '';
  function loadList() {
    return api('GET', '/api/chats' + (searchTerm ? '?q=' + encodeURIComponent(searchTerm) : '')).then(function (d) { state.conversations = d.conversations || []; renderList(); });
  }
  var searchBox = $('chatSearch'), searchTimer = null;
  if (searchBox) searchBox.addEventListener('input', function () {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(function () { searchTerm = searchBox.value.trim(); loadList(); }, 250);
  });
  els.items.addEventListener('click', function (e) {
    var del = e.target.closest('[data-del]');
    if (del) {
      if (!window.confirm('Delete this chat? This cannot be undone.')) return;
      var id = parseInt(del.getAttribute('data-del'), 10);
      api('DELETE', '/api/chats/' + id).then(function (d) {
        if (d.error) { toast(d.error); return; }
        if (state.current === id) closeThread();
        loadList();
      });
      return;
    }
    var ren = e.target.closest('[data-rename]');
    if (ren) {
      var title = window.prompt('Rename this chat', ren.getAttribute('data-title') || '');
      if (title && title.trim()) api('PATCH', '/api/chats/' + ren.getAttribute('data-rename'), { title: title.trim() }).then(function (d) {
        if (d.error) toast(d.error); else loadList();
      });
      return;
    }
    var item = e.target.closest('[data-id]');
    if (item) { openConversation(parseInt(item.getAttribute('data-id'), 10)); els.list.classList.remove('open'); }
  });
  els.toggle.addEventListener('click', function () { els.list.classList.toggle('open'); });

  /* ---------- thread ---------- */
  function closeThread() {
    state.current = null;
    els.empty.hidden = false; els.msgs.hidden = true; els.form.hidden = true; els.ctx.hidden = true;
    els.suggest.hidden = true; els.countWrap.hidden = true;
    renderList();
  }

  function renderContext(card) {
    if (!card) { els.ctx.hidden = true; return; }
    var lvl = { low: 'Low urgency', medium: 'See a doctor soon', high: 'Urgent' }[card.urgency] || '';
    els.ctx.replaceChildren(
      h('span', { class: 'muted', text: 'About your ' + (card.type === 'image' ? 'photo' : 'symptom') + ' check · ' + fmtDate(card.created_at) }),
      h('strong', { text: card.title }),
      lvl ? h('span', { class: 'badge ' + (card.urgency === 'high' ? 'badge-danger' : card.urgency === 'medium' ? 'badge-warn' : 'badge-ok'), text: lvl }) : null,
      h('a', { href: '/history' }, 'View history')
    );
    els.ctx.hidden = false;
  }

  function bubble(m) {
    var me = m.role === 'user';
    var cls = 'msg ' + (me ? 'msg-user' : 'msg-bot') + (m.flagged ? ' msg-emergency' : '');
    var children = [];
    if (m.flagged) children.push(h('div', { class: 'msg-flag' }, h('strong', { text: 'Emergency guidance' }), h('a', { class: 'btn btn-danger btn-sm', href: 'tel:' + EMERGENCY, text: 'Call ' + EMERGENCY })));
    children.push(h('div', { class: 'msg-text', text: m.content }));
    return h('div', { class: cls }, children[0], children[1]);
  }

  function renderMessages(messages, card, conv) {
    els.msgs.replaceChildren();
    messages.forEach(function (m) { els.msgs.appendChild(bubble(m)); });
    els.empty.hidden = true; els.msgs.hidden = false; els.form.hidden = false; els.countWrap.hidden = false;
    renderContext(card);
    // suggested questions only for a fresh chat
    els.suggest.replaceChildren();
    if (!messages.length) {
      var list = SUGGESTIONS[conv.source_type || 'general'];
      list.forEach(function (q) {
        els.suggest.appendChild(h('button', { type: 'button', class: 'chip', 'data-q': q, text: q }));
      });
      els.suggest.hidden = false;
    } else { els.suggest.hidden = true; }
    scrollDown();
    els.input.focus({ preventScroll: true });
  }

  function openConversation(id) {
    return api('GET', '/api/chats/' + id).then(function (d) {
      if (d.error) { toast(d.error); return; }
      state.current = id;
      renderMessages(d.messages, d.analysis, d.conversation);
      renderList();
    });
  }

  els.suggest.addEventListener('click', function (e) {
    var q = e.target.closest('[data-q]');
    if (q) sendMessage(q.getAttribute('data-q'));
  });

  /* ---------- sending ---------- */
  function sendMessage(text) {
    text = (text || '').trim();
    if (!text || state.busy || !state.current) return;
    state.busy = true; els.send.disabled = true;
    els.suggest.hidden = true;
    var optimistic = bubble({ role: 'user', content: text });
    els.msgs.appendChild(optimistic); scrollDown();
    els.input.value = ''; autosize(); updateCount();
    els.typing.hidden = false;

    api('POST', '/api/chats/' + state.current + '/messages', { message: text }).then(function (d) {
      els.typing.hidden = true;
      if (d.error) {
        optimistic.parentNode && optimistic.parentNode.removeChild(optimistic);
        els.input.value = text; updateCount();
        toast(d.error);
        return;
      }
      var reply = bubble(d.assistant_message);
      els.msgs.appendChild(reply);
      if (d.assistant_message.flagged) els.msgs.scrollTop = Math.max(0, reply.offsetTop - els.msgs.offsetTop - 8);  // emergency: show the call button first
      else scrollDown();
      loadList();
    }).catch(function (err) {
      els.typing.hidden = true;
      if (err.message !== 'auth') toast('Network problem. Check your connection and try again.');
    }).then(function () { state.busy = false; els.send.disabled = false; els.input.focus({ preventScroll: true }); });
  }

  els.form.addEventListener('submit', function (e) { e.preventDefault(); sendMessage(els.input.value); });
  els.input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(els.input.value); }
  });
  function updateCount() { els.count.textContent = String(els.input.value.length); }
  function autosize() { els.input.style.height = 'auto'; els.input.style.height = Math.min(els.input.scrollHeight, 160) + 'px'; }
  els.input.addEventListener('input', function () { updateCount(); autosize(); });

  /* ---------- new chat ---------- */
  function loadAnalyses() {
    return api('GET', '/api/analyses/recent').then(function (d) {
      els.attach.replaceChildren(h('option', { value: '', text: 'A general health question' }));
      (d.analyses || []).forEach(function (a) {
        var label = (a.type === 'image' ? 'Photo' : 'Symptoms') + ' · ' + fmtDate(a.created_at) + ' · ' + (a.title || '').slice(0, 40);
        els.attach.appendChild(h('option', { value: a.type + ':' + a.id, text: label }));
      });
    });
  }
  els.newBtn.addEventListener('click', function () { loadAnalyses(); els.panel.hidden = !els.panel.hidden; });
  els.cancel.addEventListener('click', function () { els.panel.hidden = true; });
  els.start.addEventListener('click', function () {
    var v = els.attach.value, body = {};
    if (v) { var p = v.split(':'); body = { analysis_type: p[0], analysis_id: parseInt(p[1], 10), 'new': true }; }
    api('POST', '/api/chats', body).then(function (d) {
      if (d.error) { toast(d.error); return; }
      els.panel.hidden = true; els.list.classList.remove('open');
      loadList().then(function () { state.current = d.conversation.id; renderMessages(d.messages, d.analysis, d.conversation); renderList(); });
    });
  });

  /* ---------- boot: open a chat for ?type=&id= or the latest ---------- */
  loadList().then(function () {
    var t = app.getAttribute('data-open-type'), id = parseInt(app.getAttribute('data-open-id'), 10);
    if ((t === 'image' || t === 'symptom') && id) {
      return api('POST', '/api/chats', { analysis_type: t, analysis_id: id }).then(function (d) {
        if (d.error) { toast(d.error); return; }
        return loadList().then(function () { state.current = d.conversation.id; renderMessages(d.messages, d.analysis, d.conversation); renderList(); });
      });
    }
    if (state.conversations.length) return openConversation(state.conversations[0].id);
  });
})();

/* Quick Aid - admin user management. Account status only; this script never touches anyone's health data. */
(function () {
  'use strict';
  var app = document.getElementById('adminUsersApp');
  if (!app) return;
  var doc = document;
  var CSRF = (doc.querySelector('meta[name="csrf-token"]') || {}).content || '';
  var $ = function (id) { return doc.getElementById(id); };

  function h(tag, props) {
    var el = doc.createElement(tag);
    Object.keys(props || {}).forEach(function (k) {
      if (k === 'class') el.className = props[k]; else if (k === 'text') el.textContent = props[k]; else el.setAttribute(k, props[k]);
    });
    var add = function (c) {
      if (c == null || c === false) return;
      if (Array.isArray(c)) { c.forEach(add); return; }
      el.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c);
    };
    for (var i = 2; i < arguments.length; i++) add(arguments[i]);
    return el;
  }
  function api(url, opts) {
    opts = opts || {}; opts.headers = Object.assign({ 'X-CSRFToken': CSRF }, opts.headers || {});
    return fetch(url, opts).then(function (r) { return r.json().then(function (d) { d._status = r.status; return d; }); });
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  function when(iso) { return (iso || '').replace('T', ' ').slice(0, 16); }

  var banner = $('adminUsersBanner');
  function flash(message, tone) {
    clear(banner);
    if (!message) return;
    banner.appendChild(h('div', { class: 'alert alert-' + (tone || 'info'), role: 'status' }, message));
  }

  var tbody = doc.querySelector('#usersTable tbody');
  var search = $('userSearch');
  var empty = $('usersEmpty');
  var detail = $('userDetail');
  var detailBody = $('detailBody');
  var detailEventsBody = doc.querySelector('#detailEvents tbody');

  function row(u) {
    var actions = h('div', { class: 'row-actions' },
      h('button', { type: 'button', class: 'btn btn-ghost btn-sm', 'data-act': 'detail', 'data-id': u.username }, 'Details'),
      u.totp_enabled ? h('button', { type: 'button', class: 'btn btn-ghost btn-sm', 'data-act': 'reset-2fa', 'data-id': u.username }, 'Reset 2FA') : null,
      h('button', { type: 'button', class: 'btn btn-ghost btn-sm', 'data-act': 'force-logout', 'data-id': u.username }, 'Sign out everywhere'),
      h('button', { type: 'button', class: 'btn btn-ghost btn-sm', 'data-act': 'reset-password', 'data-id': u.username }, 'Email password reset'),
      h('button', { type: 'button', class: 'btn btn-danger btn-sm', 'data-act': 'delete', 'data-id': u.username }, 'Delete'));
    return h('tr', {},
      h('td', { text: u.username }),
      h('td', { text: u.email }),
      h('td', {}, h('span', { class: 'status-dot ' + (u.email_verified ? 'on' : 'off') }), u.email_verified ? 'Yes' : 'No'),
      h('td', {}, h('span', { class: 'status-dot ' + (u.totp_enabled ? 'on' : 'off') }), u.totp_enabled ? 'On' : 'Off'),
      h('td', { text: u.is_admin ? 'Yes' : '' }),
      h('td', { text: (u.created_at || '').slice(0, 10) }),
      h('td', {}, actions));
  }

  function loadUsers(q) {
    api('/admin/api/users' + (q ? '?q=' + encodeURIComponent(q) : '')).then(function (d) {
      if (d._status !== 200) { flash(d.error || 'Could not load accounts.', 'error'); return; }
      clear(tbody);
      (d.users || []).forEach(function (u) { tbody.appendChild(row(u)); });
      empty.hidden = (d.users || []).length > 0;
      if (!d.admin_configured) flash('ADMIN_USERNAMES is not set in the environment, so this page exists but no account can sign in as an admin yet.', 'warn');
    });
  }

  function showDetail(username) {
    api('/admin/api/users/' + encodeURIComponent(username)).then(function (d) {
      if (d._status !== 200) { flash(d.error || 'Account not found.', 'error'); return; }
      var u = d.user;
      clear(detailBody);
      detailBody.appendChild(h('dl', { class: 'detail-list' },
        h('dt', { text: 'Username' }), h('dd', { text: u.username }),
        h('dt', { text: 'Email' }), h('dd', { text: u.email + (u.email_verified ? ' (verified)' : ' (not verified)') }),
        h('dt', { text: 'Two-factor authentication' }), h('dd', { text: u.totp_enabled ? ('On, ' + u.recovery_codes_remaining + ' recovery code(s) left') : 'Off' }),
        h('dt', { text: 'Account created' }), h('dd', { text: when(u.created_at) })));
      clear(detailEventsBody);
      (u.recent_security_events || []).forEach(function (e) {
        detailEventsBody.appendChild(h('tr', {}, h('td', { text: when(e.created_at) }), h('td', { text: e.event + (e.detail ? ' - ' + e.detail : '') })));
      });
      detail.hidden = false;
      detail.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
  $('closeDetail').addEventListener('click', function () { detail.hidden = true; });

  /* ---------- confirmation modal, used for every action ---------- */
  var modal = $('confirmModal'), mTitle = $('confirmTitle'), mBody = $('confirmBody'),
    mTypeBox = $('confirmTypeBox'), mType = $('confirmType'), mTypeLabel = $('confirmTypeLabel'),
    mOk = $('confirmOk'), mCancel = $('confirmCancel');

  function confirmAction(opts) {
    mTitle.textContent = opts.title;
    mBody.textContent = opts.body;
    mOk.textContent = opts.okLabel;
    mOk.className = 'btn ' + (opts.danger ? 'btn-danger' : 'btn');
    mTypeBox.hidden = !opts.typeToConfirm;
    mType.value = '';
    if (opts.typeToConfirm) mTypeLabel.textContent = 'Type "' + opts.typeToConfirm + '" to confirm';
    modal.hidden = false;
    mOk.focus();
    function cleanup() { modal.hidden = true; mOk.removeEventListener('click', onOk); mCancel.removeEventListener('click', onCancel); }
    function onOk() {
      if (opts.typeToConfirm && mType.value !== opts.typeToConfirm) { mType.focus(); return; }
      cleanup();
      opts.onConfirm();
    }
    function onCancel() { cleanup(); }
    mOk.addEventListener('click', onOk);
    mCancel.addEventListener('click', onCancel);
  }
  modal.addEventListener('keydown', function (e) { if (e.key === 'Escape') { modal.hidden = true; } });

  var ACTIONS = {
    'reset-2fa': function (username) {
      confirmAction({
        title: 'Reset two-factor authentication?', okLabel: 'Reset 2FA', danger: false,
        body: username + ' will be signed out of every device and can sign back in with just their password. They are emailed about this.',
        onConfirm: function () {
          api('/admin/api/users/' + encodeURIComponent(username) + '/reset-2fa', { method: 'POST' }).then(function (d) {
            flash(d._status === 200 ? '2FA reset for ' + username + '.' : (d.error || 'Could not reset 2FA.'), d._status === 200 ? 'success' : 'error');
            loadUsers(search.value.trim());
          });
        }
      });
    },
    'force-logout': function (username) {
      confirmAction({
        title: 'Sign out everywhere?', okLabel: 'Sign out everywhere', danger: false,
        body: 'Every device currently signed in as ' + username + ' will need to sign in again.',
        onConfirm: function () {
          api('/admin/api/users/' + encodeURIComponent(username) + '/force-logout', { method: 'POST' }).then(function (d) {
            flash(d._status === 200 ? 'Signed ' + username + ' out everywhere.' : (d.error || 'Could not sign them out.'), d._status === 200 ? 'success' : 'error');
          });
        }
      });
    },
    'reset-password': function (username) {
      confirmAction({
        title: 'Email a password-reset link?', okLabel: 'Send reset email', danger: false,
        body: username + ' will receive an email with a link to set a new password (valid 1 hour, single use).',
        onConfirm: function () {
          api('/admin/api/users/' + encodeURIComponent(username) + '/send-password-reset', { method: 'POST' }).then(function (d) {
            flash(d._status === 200 ? 'Password-reset email sent to ' + username + '.' : (d.error || 'Could not send the reset email.'), d._status === 200 ? 'success' : 'error');
          });
        }
      });
    },
    'delete': function (username) {
      confirmAction({
        title: 'Permanently delete this account?', okLabel: 'Delete account', danger: true, typeToConfirm: username,
        body: 'This deletes ' + username + ' and everything they own. This cannot be undone.',
        onConfirm: function () {
          api('/admin/api/users/' + encodeURIComponent(username) + '/delete', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirm: username })
          }).then(function (d) {
            flash(d._status === 200 ? 'Deleted ' + username + '.' : (d.error || 'Could not delete the account.'), d._status === 200 ? 'success' : 'error');
            if (d._status === 200) { detail.hidden = true; loadUsers(search.value.trim()); }
          });
        }
      });
    },
    'detail': function (username) { showDetail(username); }
  };

  tbody.addEventListener('click', function (e) {
    var btn = e.target.closest('button[data-act]');
    if (!btn) return;
    var fn = ACTIONS[btn.getAttribute('data-act')];
    if (fn) fn(btn.getAttribute('data-id'));
  });

  var searchTimer;
  search.addEventListener('input', function () {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(function () { loadUsers(search.value.trim()); }, 250);
  });

  loadUsers('');
})();

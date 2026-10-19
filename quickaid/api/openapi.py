"""OpenAPI 3.0 description, generated from the live routing table.

Each documented view carries a `@doc(...)` decorator right next to its code. `build_spec(app)` walks the app's real URL map, so a
route cannot be documented with the wrong path or method, and the request-body schemas come straight from the same Pydantic models
the code validates with. A test fails if any JSON API route is missing its `@doc`.
"""

import re
from typing import Any, Dict, List, Optional

from quickaid.api import schemas
from quickaid.api.responses import DEFAULT_CODES

API_VERSION = '1.0.0'
_CONVERTER = re.compile(r'<(?:(?P<conv>[a-z_]+)(?:\((?P<args>[^)]*)\))?:)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)>')


def doc(summary: str, *, tags: Optional[List[str]] = None, auth: bool = True, admin: bool = False, request: Optional[str] = None,
        query: Optional[str] = None, description: str = '', responses: Optional[Dict[int, str]] = None, hidden: bool = False,
        download: Optional[str] = None):
    """Attach documentation to a view. `request`/`query` are names of Pydantic models in quickaid.api.schemas."""
    def deco(fn):
        fn.__api_doc__ = {'summary': summary, 'tags': tags or ['General'], 'auth': auth, 'admin': admin, 'request': request, 'query': query,
                          'description': description, 'responses': responses or {}, 'hidden': hidden, 'download': download}
        return fn
    return deco


def _oas30(node: Any) -> Any:
    """Pydantic emits JSON-Schema 2020-12 (`anyOf: [X, {type: null}]`, `const`); OpenAPI 3.0 spells those `nullable: true` and `enum`."""
    if isinstance(node, list):
        return [_oas30(x) for x in node]
    if not isinstance(node, dict):
        return node
    node = {k: _oas30(v) for k, v in node.items()}
    if 'const' in node:
        node['enum'] = [node.pop('const')]
    options = node.get('anyOf')
    if options:
        rest = [o for o in options if o != {'type': 'null'}]
        if len(rest) != len(options):
            node['nullable'] = True
            node.pop('anyOf')
            if len(rest) == 1:
                node.update(rest[0])
            else:
                node['anyOf'] = rest
    return node


def _model_schema(name: str, components: Dict) -> Dict:
    model = getattr(schemas, name)
    js = model.model_json_schema(ref_template='#/components/schemas/{model}')
    for def_name, d in (js.pop('$defs', {}) or {}).items():
        components[def_name] = _oas30(d)
    components[name] = _oas30(js)
    return {'$ref': f'#/components/schemas/{name}'}


def _openapi_path(rule: str):
    params = []

    def sub(m):
        conv, args, name = m.group('conv'), m.group('args'), m.group('name')
        schema: Dict[str, Any] = {'type': 'integer'} if conv == 'int' else {'type': 'string'}
        if conv == 'any' and args:
            schema['enum'] = [a.strip() for a in args.split(',')]
        params.append({'name': name, 'in': 'path', 'required': True, 'schema': schema})
        return '{' + name + '}'
    return _CONVERTER.sub(sub, rule), params


def _query_params(model_name: str) -> List[Dict]:
    out = []
    model = getattr(schemas, model_name)
    for name, field in model.model_fields.items():
        js = model.model_json_schema()['properties'].get(name, {})
        js = _oas30({k: v for k, v in js.items() if k not in ('title', 'description')})
        out.append({'name': name, 'in': 'query', 'required': False, 'description': field.description or '', 'schema': js or {'type': 'string'},
                    **({'style': 'form', 'explode': False} if js.get('type') == 'array' else {})})
    return out


def build_spec(app) -> Dict:
    components: Dict[str, Any] = {
        'Error': {'type': 'object', 'required': ['success', 'error', 'code'], 'properties': {
            'success': {'type': 'boolean', 'enum': [False]},
            'error': {'type': 'string', 'description': 'Human-readable message.'},
            'code': {'type': 'string', 'description': 'Stable machine-readable code, e.g. entry_not_found, validation_error.'},
            'request_id': {'type': 'string', 'description': 'Also in the X-Request-ID header and the server log.'},
            'hints': {'type': 'array', 'items': {'type': 'string'}, 'description': 'What to do next.'},
            'details': {'type': 'array', 'items': {'type': 'object', 'properties': {'field': {'type': 'string'}, 'message': {'type': 'string'}}},
                        'description': 'Per-field validation problems.'}}},
        'Success': {'type': 'object', 'required': ['success'], 'additionalProperties': True, 'properties': {
            'success': {'type': 'boolean', 'enum': [True]}, 'request_id': {'type': 'string'}},
            'description': 'The payload fields sit at the top level next to `success`.'},
    }
    paths: Dict[str, Any] = {}
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        fn = app.view_functions.get(rule.endpoint)
        d = getattr(fn, '__api_doc__', None)
        if not d or d['hidden']:
            continue
        path, path_params = _openapi_path(rule.rule)
        for method in sorted(rule.methods - {'HEAD', 'OPTIONS'}):
            responses = {str(c): {'description': txt, 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Success'}}}}
                         for c, txt in d['responses'].items() if c < 400}
            if not responses:
                responses = {'200': {'description': 'Success', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Success'}}}}}
            if d['download']:
                responses = {'200': {'description': 'A file download', 'content': {d['download']: {'schema': {'type': 'string', 'format': 'binary'}}}}}
            for code in sorted({400, 404, 429, 500} | {c for c in d['responses'] if c >= 400} | ({401} if d['auth'] else set())):
                responses.setdefault(str(code), {'description': d['responses'].get(code, DEFAULT_CODES.get(code, 'Error').replace('_', ' ').capitalize()),
                                                 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Error'}}}})
            op: Dict[str, Any] = {'operationId': f"{rule.endpoint.replace('.', '_')}_{method.lower()}", 'summary': d['summary'], 'tags': d['tags'], 'responses': responses}
            if d['description']:
                op['description'] = d['description']
            params = list(path_params) + (_query_params(d['query']) if d['query'] else [])
            if params:
                op['parameters'] = params
            if d['request'] and method in ('POST', 'PUT', 'PATCH'):
                op['requestBody'] = {'required': True, 'content': {'application/json': {'schema': _model_schema(d['request'], components)}}}
            if d['auth']:
                unsafe = method in ('POST', 'PUT', 'PATCH', 'DELETE')
                op['security'] = [{'cookieAuth': [], 'csrfToken': []} if unsafe else {'cookieAuth': []}]
            if d['admin']:
                op['description'] = (op.get('description', '') + ' Admins only (ADMIN_USERNAMES); other users receive 404.').strip()
            paths.setdefault(path, {})[method.lower()] = op
    return {
        'openapi': '3.0.3',
        'info': {'title': 'Quick Aid API', 'version': API_VERSION,
                 'description': ('JSON API behind the Quick Aid web app. Authentication is the signed-in session cookie; every state-changing request must also '
                                 'send the CSRF token in the `X-CSRFToken` header (it is in the page\'s `csrf-token` meta tag). All errors share one shape '
                                 '(see `Error`). Everything here is AI-generated information, not medical advice; call your local emergency number in an emergency.')},
        'servers': [{'url': '/'}],
        'tags': [{'name': t} for t in ('Analysis', 'Interview', 'Conversations', 'History', 'Privacy', 'Account', 'Admin', 'System')],
        'paths': paths,
        'components': {'schemas': components, 'securitySchemes': {
            'cookieAuth': {'type': 'apiKey', 'in': 'cookie', 'name': 'session', 'description': 'Flask session cookie set by logging in.'},
            'csrfToken': {'type': 'apiKey', 'in': 'header', 'name': 'X-CSRFToken', 'description': 'CSRF token for state-changing requests.'}}},
    }

"""Follow-up conversations."""

from flask import Blueprint, Response, render_template, request
from flask_login import current_user, login_required

from quickaid import config
from quickaid.api.openapi import doc
from quickaid.api.responses import ok
from quickaid.api.schemas import ConversationQuery, RenameBody
from quickaid.api.validation import analysis_ref, chat_message, json_object, parse_model, parse_query
from quickaid.extensions import limiter, user_or_ip
from quickaid.services.container import get_services

bp = Blueprint('conversations', __name__)


def _uid() -> int:
    return int(current_user.id)


@bp.route('/chat')
@login_required
def chat_page():
    return render_template('chat.html', active_page='chat', open_type=request.args.get('type', ''), open_id=request.args.get('id', ''),
                           max_chat_length=config.MAX_CHAT_MESSAGE_LENGTH, ai_ready=get_services().chat_ai.use_gemini)


@bp.route('/api/chats', methods=['GET'])
@login_required
@doc('List or search your conversations', tags=['Conversations'], query='ConversationQuery',
     description='Newest first. `q` matches titles and message text. Paginated.')
def list_chats():
    q = parse_query(ConversationQuery)
    res = get_services().chats.list(_uid(), q.q, q.page, q.per_page)
    return ok({'conversations': res['items'], 'pagination': {k: res[k] for k in ('page', 'per_page', 'total', 'pages')}})


@bp.route('/api/analyses/recent')
@login_required
@doc('Your most recent assessments (for attaching to a new chat)', tags=['Conversations'])
def recent_analyses():
    return ok({'analyses': get_services().chats.recent_analyses(_uid())})


@bp.route('/api/chats', methods=['POST'])
@login_required
@limiter.limit("20 per minute", key_func=user_or_ip)
@doc('Start a chat about one of your assessments, or a general chat', tags=['Conversations'], request='ChatCreateRequest',
     responses={201: 'Created', 404: 'No such assessment (or it is not yours)'})
def create_chat():
    data = json_object()
    atype = aid = None
    if data.get('analysis_type') or data.get('analysis_id'):
        atype, aid = analysis_ref(data)
    out = get_services().chats.create(_uid(), atype, aid, bool(data.get('new')))
    return ok(out, 201 if out['created'] else 200)


@bp.route('/api/chats/<int:conversation_id>')
@login_required
@doc('Open one conversation with its messages', tags=['Conversations'], responses={404: 'No such conversation (or it is not yours)'})
def get_chat(conversation_id):
    return ok(get_services().chats.get(_uid(), conversation_id))


@bp.route('/api/chats/<int:conversation_id>/messages', methods=['POST'])
@login_required
@limiter.limit("20 per minute", key_func=user_or_ip)
@doc('Send a message and get the reply', tags=['Conversations'], request='ChatMessageRequest',
     description='Emergency wording is answered immediately with guidance, without asking the AI.',
     responses={404: 'No such conversation (or it is not yours)'})
def post_chat_message(conversation_id):
    message = chat_message(json_object().get('message'))
    return ok(get_services().chats.send(_uid(), conversation_id, message))


@bp.route('/api/chats/<int:conversation_id>', methods=['PATCH'])
@login_required
@limiter.limit("30 per minute", key_func=user_or_ip)
@doc('Rename a conversation', tags=['Conversations'], request='RenameBody', responses={404: 'No such conversation (or it is not yours)'})
def rename_chat(conversation_id):
    body = parse_model(RenameBody, json_object())
    return ok(get_services().chats.rename(_uid(), conversation_id, body.title))


@bp.route('/api/chats/<int:conversation_id>', methods=['DELETE'])
@login_required
@limiter.limit("20 per minute", key_func=user_or_ip)
@doc('Delete a conversation', tags=['Conversations'], responses={404: 'No such conversation (or it is not yours)'})
def delete_chat(conversation_id):
    get_services().chats.delete(_uid(), conversation_id)
    return ok()


@bp.route('/api/chats/<int:conversation_id>/export')
@login_required
@limiter.limit("20 per hour", key_func=user_or_ip)
@doc('Download one conversation (?format=json or txt)', tags=['Conversations'], download='application/json',
     responses={404: 'No such conversation (or it is not yours)'})
def export_chat(conversation_id):
    fmt = 'txt' if request.args.get('format') == 'txt' else 'json'
    body, mimetype, filename = get_services().chats.export(_uid(), conversation_id, fmt)
    return Response(body, mimetype=mimetype, headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@bp.route('/api/follow_up', methods=['POST'])
@login_required
@limiter.limit("20 per minute", key_func=user_or_ip)
@doc('Ask one follow-up question about an assessment', tags=['Conversations'], request='FollowUpRequest',
     description='Finds or creates the chat for that assessment, so the exchange also appears on the Chat page.',
     responses={404: 'No such assessment (or it is not yours)'})
def follow_up():
    data = json_object()
    atype, aid = analysis_ref(data)
    question = chat_message(data.get('question'))
    return ok(get_services().chats.follow_up(_uid(), atype, aid, question))

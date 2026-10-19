"""Follow-up conversations: creating, reading, sending, searching, renaming, exporting and deleting. Always scoped to one user."""

import json
from typing import Dict, Optional

from quickaid import config
from quickaid.ai.conversation import ConversationService as ChatAI, summarize_analysis
from quickaid.api.responses import ApiError
from quickaid.core import database as db, history_repo
from quickaid.core.localization import get_app_region
from quickaid.core.logging_config import get_logger

logger = get_logger('chat_service')


class ChatService:
    def __init__(self, chat_ai: ChatAI):
        self.chat_ai = chat_ai

    # ---- helpers
    @staticmethod
    def analysis_title(entry: Dict) -> str:
        if entry['type'] == 'image':
            return (entry['detected_conditions'][0] if entry.get('detected_conditions') else entry.get('original_filename')) or 'Photo check'
        return (entry['possible_conditions'][0] if entry.get('possible_conditions') else entry.get('symptom_text', '')) or 'Symptom check'

    def analysis_card(self, entry: Dict) -> Dict:
        level = entry.get('urgency') if entry['type'] == 'image' else entry.get('urgency_level')
        return {'type': entry['type'], 'id': entry['id'], 'title': self.analysis_title(entry)[:80], 'created_at': entry['created_at'], 'urgency': level}

    def _owned_conversation(self, user_id: int, conversation_id: int) -> Dict:
        conv = db.get_conversation(user_id, conversation_id)
        if conv is None:                      # 404 for "not yours" and "does not exist" alike: ids cannot be probed
            raise ApiError('Conversation not found', 404, 'conversation_not_found')
        return conv

    # ---- reads
    def list(self, user_id: int, q: str = '', page: int = 1, per_page: int = 50) -> Dict:
        return history_repo.search_conversations(user_id, q, page, per_page)

    def recent_analyses(self, user_id: int) -> list:
        return db.list_recent_analyses(user_id, 20)

    def get(self, user_id: int, conversation_id: int) -> Dict:
        conv = self._owned_conversation(user_id, conversation_id)
        card = None
        if conv['source_type']:
            entry = db.get_analysis(user_id, conv['source_type'], conv['source_id'])
            card = self.analysis_card(entry) if entry else None
        return {'conversation': conv, 'messages': db.get_messages(conversation_id), 'analysis': card}

    # ---- writes
    def create(self, user_id: int, analysis_type: Optional[str], analysis_id: Optional[int], new: bool = False) -> Dict:
        """Start a chat about one of the user's analyses (reusing the existing one), or a general chat."""
        if analysis_type:
            entry = db.get_analysis(user_id, analysis_type, analysis_id)
            if entry is None:
                raise ApiError('Analysis not found', 404, 'analysis_not_found')
            existing = None if new else db.find_conversation_for_analysis(user_id, analysis_type, analysis_id)
            if existing:
                return {'created': False, 'conversation': existing, 'messages': db.get_messages(existing['id']), 'analysis': self.analysis_card(entry)}
            conv = db.create_conversation(user_id, self.analysis_title(entry), analysis_type, analysis_id)
            return {'created': True, 'conversation': conv, 'messages': [], 'analysis': self.analysis_card(entry)}
        return {'created': True, 'conversation': db.create_conversation(user_id, 'New chat'), 'messages': [], 'analysis': None}

    def _send(self, user_id: int, conversation: Dict, message: str) -> Dict:
        """Store the user's message, get a reply (safety checks happen inside), store it."""
        if conversation['message_count'] >= db.MAX_MESSAGES_PER_CONVERSATION:
            raise ApiError('This chat is full. Please start a new chat.', 400, 'chat_full')
        summary, atype = '{}', None
        if conversation['source_type']:
            entry = db.get_analysis(user_id, conversation['source_type'], conversation['source_id'])
            if entry:
                atype, summary = conversation['source_type'], summarize_analysis(conversation['source_type'], entry)
        history = db.get_messages(conversation['id'], limit=20)
        user_msg = db.add_message(conversation['id'], 'user', message)
        result = self.chat_ai.reply(atype, summary, history, message, region=get_app_region())
        bot_msg = db.add_message(conversation['id'], 'model', result['answer'], flagged=result['flagged'])
        if conversation['title'] == 'New chat':
            db.rename_conversation(user_id, conversation['id'], message)
        return {'user_message': user_msg, 'assistant_message': bot_msg, 'mode': result['mode'], 'emergency': result['emergency']}

    def send(self, user_id: int, conversation_id: int, message: str) -> Dict:
        conv = self._owned_conversation(user_id, conversation_id)
        try:
            return self._send(user_id, conv, message)
        except ApiError:
            raise
        except Exception as e:
            logger.error("Chat message failed for user_id=%s", user_id, exc_info=True)
            raise ApiError(f'Could not send that message: {e}', 500, 'chat_failed')

    def follow_up(self, user_id: int, analysis_type: str, analysis_id: int, question: str) -> Dict:
        entry = db.get_analysis(user_id, analysis_type, analysis_id)
        if entry is None:
            raise ApiError('Analysis not found', 404, 'analysis_not_found')
        conv = db.find_conversation_for_analysis(user_id, analysis_type, analysis_id) or (
            db.create_conversation(user_id, self.analysis_title(entry), analysis_type, analysis_id) | {'message_count': 0})
        try:
            sent = self._send(user_id, conv, question)
        except ApiError:
            raise
        except Exception as e:
            logger.error("Follow-up failed for user_id=%s", user_id, exc_info=True)
            raise ApiError(f'Could not process that question: {e}', 500, 'chat_failed')
        history = [{'role': m['role'], 'content': m['content']} for m in db.get_messages(conv['id'])]
        return {'answer': sent['assistant_message']['content'], 'conversation_id': conv['id'], 'follow_up_history': history, 'mode': sent['mode']}

    def rename(self, user_id: int, conversation_id: int, title: str) -> Dict:
        self._owned_conversation(user_id, conversation_id)
        db.rename_conversation(user_id, conversation_id, title)
        return {'conversation': db.get_conversation(user_id, conversation_id)}

    def delete(self, user_id: int, conversation_id: int) -> None:
        if not db.delete_conversation(user_id, conversation_id):
            raise ApiError('Conversation not found', 404, 'conversation_not_found')

    def export(self, user_id: int, conversation_id: int, fmt: str = 'json'):
        """(bytes, mimetype, filename) for one conversation."""
        data = self.get(user_id, conversation_id)
        conv = data['conversation']
        if fmt == 'txt':
            lines = [conv['title'], f"Started {conv['created_at']}", 'AI-generated information, not medical advice.', '']
            lines += [f"[{m['created_at'][:16]}] {'You' if m['role'] == 'user' else 'Quick Aid'}: {m['content']}\n" for m in data['messages']]
            return '\n'.join(lines).encode('utf-8'), 'text/plain; charset=utf-8', f'quickaid-chat-{conversation_id}.txt'
        return (json.dumps({'conversation': conv, 'messages': data['messages']}, indent=1, ensure_ascii=False).encode('utf-8'),
                'application/json', f'quickaid-chat-{conversation_id}.json')

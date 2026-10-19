"""Public and landing pages."""

from flask import Blueprint, render_template
from flask_login import current_user

from quickaid.services.container import get_services

bp = Blueprint('pages', __name__)


@bp.route('/')
def index():
    if not current_user.is_authenticated:
        return render_template('index.html')
    stats, recent = get_services().history.dashboard(int(current_user.id))
    return render_template('index.html', stats=stats, recent=recent)


@bp.route('/emergency')
def emergency():
    return render_template('emergency.html')


@bp.route('/privacy')
def privacy():
    """Plain-language statement of what is stored, for how long, and what you can do about it (generated from the live policy)."""
    return render_template('privacy.html', policy=get_services().retention.policy(), active_page='privacy')

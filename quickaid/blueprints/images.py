"""Photo analysis."""

from flask import Blueprint, current_app, request
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from quickaid.api.openapi import doc
from quickaid.api.responses import ok
from quickaid.extensions import limiter
from quickaid.services.container import get_services

bp = Blueprint('images', __name__)


@bp.route('/upload', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
@doc('Analyse one photo (multipart/form-data, field "file")', tags=['Analysis'],
     description=('The file is judged by its real content, analysed, and DELETED immediately, even if analysis fails. Problems come back with a '
                  'specific `code` (empty_file, too_large, heic_unsupported, not_an_image, corrupt, too_small, too_many_pixels, ...) and `hints`.'),
     responses={413: 'Larger than 16 MB'})
def upload_file():
    file = request.files.get('file')
    data = file.read() if file is not None and file.filename else None
    name = secure_filename(file.filename) if file is not None and file.filename else ''
    result = get_services().analysis.analyze_image(int(current_user.id), data, file.filename if file is not None else '',
                                                   current_app.config['UPLOAD_FOLDER'], name or 'photo')
    return ok(result)

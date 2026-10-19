"""Server-side validation of an uploaded photo, with specific, friendly error messages.

The browser checks the same things first for instant feedback, but the browser is never trusted: this runs on
every upload and decides from the file's real CONTENT, not its name or the type the browser claims.

Each problem has a stable `code` (for the UI and tests), a plain-language `message`, and `hints` that tell the
person exactly what to do next.
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

MAX_BYTES = 16 * 1024 * 1024
MIN_SIDE = 80                   # px: below this nothing can be assessed
MAX_PIXELS = 40_000_000         # guards against decompression bombs (a tiny file that expands to gigabytes)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
SUPPORTED_TEXT = 'JPG, PNG, WebP, GIF or BMP'

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


@dataclass
class UploadProblem:
    code: str
    message: str
    hints: List[str] = field(default_factory=list)
    status: int = 400


@dataclass
class ValidatedImage:
    ext: str
    width: int
    height: int
    size: int
    frames: int = 1


def sniff_format(head: bytes) -> Optional[str]:
    """The real format from the first bytes (never from the file name)."""
    if head.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if head.startswith(b'\xff\xd8\xff'):
        return 'jpg'
    if head[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    if head.startswith(b'BM'):
        return 'bmp'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'webp'
    if head[4:8] == b'ftyp' and head[8:12] in (b'heic', b'heix', b'hevc', b'heim', b'heis', b'mif1', b'msf1', b'avif'):
        return 'heic'
    return None


def _mb(n: int) -> str:
    return f'{n / 1048576:.1f} MB'


def validate_upload(data: bytes, filename: str = '') -> Tuple[Optional[ValidatedImage], Optional[UploadProblem]]:
    """Return (image_info, None) when the upload is a usable photo, otherwise (None, problem)."""
    if data is None:
        return None, UploadProblem('no_file', 'No photo was selected.', ['Choose a photo, or take one with the camera.'])
    if len(data) == 0:
        return None, UploadProblem('empty_file', 'That file is empty (0 bytes).', ['Choose the photo again, or take a new one.'])
    if len(data) > MAX_BYTES:
        return None, UploadProblem('too_large', f'That photo is {_mb(len(data))}, and the limit is {_mb(MAX_BYTES)}.',
                                   ['Use the Take photo button, which makes a smaller file.',
                                    'Or choose a lower-resolution copy of the photo.'], 413)

    ext_in_name = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    real = sniff_format(data[:32])
    if real == 'heic':
        return None, UploadProblem('heic_unsupported', "That looks like an iPhone HEIC/HEIF photo, which can't be read yet.",
                                   ['On iPhone: Settings → Camera → Formats → choose "Most Compatible".',
                                    'Or use the Take photo button here, which saves a JPG.'])
    if real is None:
        if ext_in_name and ext_in_name not in ALLOWED_EXTENSIONS:
            return None, UploadProblem('unsupported_type', f"Files of type .{ext_in_name} can't be used. Supported: {SUPPORTED_TEXT}.",
                                       ['If this is a document or screenshot, take a photo of the affected area instead.'])
        return None, UploadProblem('not_an_image', f"That file isn't a photo we can read. Supported: {SUPPORTED_TEXT}.",
                                   ['Take a photo of the affected area with your camera.',
                                    'Documents, PDFs and screenshots cannot be analysed.'])

    try:
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
            frames = getattr(im, 'n_frames', 1)
            if width * height > MAX_PIXELS:
                return None, UploadProblem('too_many_pixels', f'This image is extremely large ({width * height / 1e6:.0f} megapixels).',
                                           ['Take a normal photo with the camera and upload that.'])
            im.verify()
        with Image.open(BytesIO(data)) as im2:      # verify() leaves the image unusable, so decode again from scratch
            im2.load()
    except Image.DecompressionBombError:
        return None, UploadProblem('too_many_pixels', 'This image is far too large to process.', ['Take a normal photo with the camera and upload that.'])
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError):
        return None, UploadProblem('corrupt', 'This image file looks damaged or incomplete.',
                                   ['Choose the photo again, or take a new photo.', 'If it came from a message app, try saving it to your phone first.'])

    if min(width, height) < MIN_SIDE:
        return None, UploadProblem('too_small', f'The photo is only {width}×{height} pixels, which is too small to see any detail.',
                                   ['Take the photo at full size (not a thumbnail or a tiny crop).',
                                    'Move closer so the affected area fills the frame.'])
    return ValidatedImage(ext=real, width=width, height=height, size=len(data), frames=frames), None

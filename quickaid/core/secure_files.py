"""Best-effort secure removal of temporary files (uploaded photos)."""

import os


def secure_remove(path: str) -> bool:
    """Overwrite a file with zeros, flush it to disk, then unlink it. Returns True if the file existed and is gone.

    Honest limits: on SSDs and copy-on-write filesystems an overwrite cannot be guaranteed to reach the original blocks, so this
    is defence in depth, not a promise. The real protections are that photos are short-lived (deleted right after analysis) and
    that the disk should be encrypted at the host level.
    """
    try:
        size = os.path.getsize(path)
        with open(path, 'r+b', buffering=0) as f:
            remaining, chunk = size, b'\x00' * 65536
            while remaining > 0:
                f.write(chunk[:remaining])
                remaining -= len(chunk)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass                        # even if overwriting failed, still try to remove it
    try:
        os.remove(path)
        return True
    except OSError:
        return False

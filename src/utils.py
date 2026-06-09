"""Utility functions — header decoding, timestamp parsing, helpers."""

import os
from datetime import datetime, timezone
from email.header import decode_header
from email.utils import getaddresses, parsedate_to_datetime


def decode_header_value(value: str | None) -> str:
    """Decode an RFC 2047 encoded-word header value to a plain string."""
    if not value:
        return ""
    try:
        parts = decode_header(value)
        decoded = []
        for fragment, charset in parts:
            if isinstance(fragment, bytes):
                if charset:
                    decoded.append(fragment.decode(charset, errors="replace"))
                else:
                    decoded.append(_decode_unknown_bytes(fragment))
            else:
                decoded.append(fragment)
        return "".join(decoded)
    except Exception:
        return str(value)


def _decode_unknown_bytes(data: bytes) -> str:
    """Decode bytes with unknown charset: try UTF-8 strictly, then Windows-1250."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("windows-1250")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def parse_address_list(header_value: str | None) -> list[dict]:
    """Parse a To/Cc/From header into a list of {"name": ..., "address": ...} dicts."""
    if not header_value:
        return []
    try:
        decoded = decode_header_value(header_value)
        pairs = getaddresses([decoded])
        result = []
        for name, address in pairs:
            if address:
                result.append({"name": name.strip(), "address": address.strip().lower()})
        return result
    except Exception:
        return []


def parse_date(header_value: str | None) -> str | None:
    """Parse an email Date header to an ISO 8601 string in UTC, or None on failure."""
    if not header_value:
        return None
    try:
        dt = parsedate_to_datetime(header_value.strip())
        # normalize to UTC
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def extract_attachments(message) -> tuple[list[str], list[str]]:
    """Return (names, extensions) lists for all attachment parts in a Message object."""
    names: list[str] = []
    extensions: list[str] = []
    if message is None:
        return names, extensions
    try:
        for part in message.walk():
            disposition = part.get_content_disposition()
            if disposition not in ("attachment", "inline"):
                continue
            filename = part.get_filename()
            if not filename:
                continue
            filename = decode_header_value(filename)
            names.append(filename)
            _, ext = os.path.splitext(filename)
            extensions.append(ext.lstrip(".").lower() if ext else "")
    except Exception:
        pass
    return names, extensions


def extract_thread_id(
    message_id: str | None,
    in_reply_to: str | None,
    references: str | None,
) -> str:
    """Return the root Message-ID of the thread.

    Walks References (oldest first), then falls back to In-Reply-To, then to
    message_id itself.  Always returns a non-empty string.
    """
    def clean(value: str | None) -> str:
        return (value or "").strip()

    if references:
        # References lists IDs from oldest to newest separated by whitespace
        ids = clean(references).split()
        if ids:
            return ids[0]

    reply_to = clean(in_reply_to)
    if reply_to:
        return reply_to

    mid = clean(message_id)
    return mid if mid else ""

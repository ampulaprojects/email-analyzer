"""
Email type classifier — WORK / BULK / SOCIAL.

Single responsibility: given one email dict (with keys from_address, subject),
return EmailType. No LLM, no DB — pure regex rules.

Priority: BULK > SOCIAL > WORK (default).
When in doubt → WORK (we'd rather see noise than lose real work).

Usage:
    from src.email_filter import classify_email_type, EmailType
    typ = classify_email_type(email_dict)   # email_dict has from_address, subject
"""

import re
from enum import Enum


class EmailType(str, Enum):
    WORK   = "WORK"
    BULK   = "BULK"
    SOCIAL = "SOCIAL"


# ── BULK: automated senders ────────────────────────────────────────────────────

_BULK_FROM = re.compile(
    r"(no-reply|noreply|do-not-reply|donotreply|"
    r"notifications?|mailer-?daemon|automated|"
    r"dalux\.com|asana\.com|zoomsphere|autodesk\.com|wetransfer\.com)",
    re.IGNORECASE,
)

# ── BULK: auto-generated subjects ─────────────────────────────────────────────

_BULK_SUBJ = re.compile(
    r"(^meeting\s+minutes\s*[:/]|"          # Dalux "Meeting minutes: ..."
    r"^meeting\s+invitation\s*[:/]|"        # Dalux/calendar "Meeting invitation: ..."
    r"\byou\s+have\s+been\s+mentioned\b|"   # Asana/project tools
    r"^recall\s*[:/]|"                      # Outlook recall "Recall: ..."
    r"\bP0080\b|"                           # Dalux project code auto-msgs
    r"has\s+\d+\s+updates?\s+since\b|"     # Asana "Task has 3 updates since"
    r"automatick[áa]\s+správ|"             # "automatická správa" — Slovak
    r"\bnewsletter\b|\bunsubscribe\b|"
    r"\bweekly\s+report\b|"
    r"\bscheduler\b)",
    re.IGNORECASE,
)

# ── SOCIAL: purely social events ───────────────────────────────────────────────
# Be conservative — only match when the social event IS the whole point.
# "stretnutie" (meeting) alone is NOT social — could be a project meeting.

_SOCIAL_SUBJ = re.compile(
    r"(\bparty\b|"
    r"\bgrill\b|\bbbq\b|\bbarbecue\b|"
    r"\bbeach\b|"
    r"\bvedro\b|"                  # "vedro" = bucket (Slovak slang for beach party kit)
    r"\bchill\b|"
    r"\bvolejbal\b|"               # volleyball
    r"\boslava\b|"                 # celebration/party
    r"\bnarodeninov|narodeninám\b|" # birthday (narodeninová, narodeninám...)
    r"\bnaroden[iy]\b)",           # "narodeniny" = birthday
    re.IGNORECASE,
)


def classify_email_type(email: dict) -> EmailType:
    """
    Classify a single email dict as WORK / BULK / SOCIAL.

    Checks from_address and subject only — no body parsing, no LLM.
    Returns WORK when uncertain (conservative: preserve real work).
    """
    from_addr = (email.get("from_address") or "").lower()
    subject   = email.get("subject") or ""

    if _BULK_FROM.search(from_addr) or _BULK_SUBJ.search(subject):
        return EmailType.BULK

    if _SOCIAL_SUBJ.search(subject):
        return EmailType.SOCIAL

    return EmailType.WORK

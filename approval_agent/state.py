"""
ApprovalState — the shared state schema used across all LangGraph nodes.
"""

from typing import TypedDict


class ApprovalState(TypedDict):
    script_id: str
    script_text: str
    version: int
    client_name: str
    client_email: str
    client_channel: str        # always "email" for now
    sla_hours: int             # default 48
    sent_at: str               # ISO-8601 timestamp
    follow_ups_sent: int       # 0, 1, or 2
    status: str                # pending | approved | revision | rejected | escalated | paused
    response_raw: str          # raw client reply text
    revision_notes: str        # extracted structured notes
    account_manager_email: str

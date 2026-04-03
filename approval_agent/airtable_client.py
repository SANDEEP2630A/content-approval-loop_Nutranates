"""
airtable_client.py — Read/write helpers for the Scripts table in Airtable.
"""

import os
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from pyairtable import Api

load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY", "")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Scripts")


def _get_table():
    """Return a pyairtable Table object."""
    api = Api(AIRTABLE_API_KEY)
    return api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)


# ─── Read ────────────────────────────────────────────────────────────────────


def get_script_record(script_id: str) -> Optional[dict[str, Any]]:
    """
    Fetch a single record from Airtable by its Script ID field.

    Returns the full record dict (with 'id' and 'fields') or None.
    """
    table = _get_table()
    formula = f"{{Script ID}} = '{script_id}'"
    records = table.all(formula=formula)
    if records:
        return records[0]
    return None


def get_script_fields(script_id: str) -> Optional[dict[str, Any]]:
    """
    Convenience wrapper — returns only the 'fields' portion of a record.
    """
    record = get_script_record(script_id)
    if record:
        return record["fields"]
    return None


# ─── Write ───────────────────────────────────────────────────────────────────


def update_script_fields(script_id: str, updates: dict[str, Any]) -> bool:
    """
    Update one or more fields of a record identified by Script ID.

    Args:
        script_id: e.g. "SCR-001"
        updates: dict of field names → new values

    Returns:
        True on success, False otherwise.
    """
    record = get_script_record(script_id)
    if not record:
        print(f"[AIRTABLE] Record {script_id} not found.")
        return False

    table = _get_table()
    try:
        table.update(record["id"], updates)
        print(f"[AIRTABLE] Updated {script_id}: {list(updates.keys())}")
        return True
    except Exception as exc:
        print(f"[AIRTABLE ERROR] Failed to update {script_id}: {exc}")
        return False


def set_status(script_id: str, status: str) -> bool:
    """Shortcut to update only the Status field."""
    return update_script_fields(script_id, {"Status": status})


def set_sent_at(script_id: str) -> bool:
    """Set the Sent At field to now (ISO-8601 date)."""
    now = datetime.utcnow().strftime("%Y-%m-%d")
    return update_script_fields(script_id, {"Sent At": now})


def increment_follow_ups(script_id: str, current_count: int) -> bool:
    """Increment Follow Ups Sent by 1."""
    return update_script_fields(
        script_id, {"Follow Ups Sent": current_count + 1}
    )


def save_revision_notes(script_id: str, notes: str) -> bool:
    """Persist revision notes to Airtable."""
    return update_script_fields(script_id, {"Revision Notes": notes})

"""
graph.py — LangGraph state machine for the Script Approval workflow.

Nodes:
    send_approval, wait_for_response, classify_response,
    send_followup_1, send_followup_2, escalate,
    process_approved, process_revision, process_rejected,
    handle_call_request

Conditional edges route based on response availability and classification.
"""

from datetime import datetime, timezone

from langgraph.graph import END, StateGraph

from airtable_client import (
    increment_follow_ups,
    save_revision_notes,
    set_sent_at,
    set_status,
    update_script_fields,
)
from chains import classify_response, extract_revision_notes, format_approval_email
from email_client import send_email
from state import ApprovalState


# ═══════════════════════════════════════════════════════════════════════════════
# Node functions
# ═══════════════════════════════════════════════════════════════════════════════


def send_approval(state: ApprovalState) -> dict:
    """Format an approval email via Chain 1, send it, and update Airtable."""
    email_body = format_approval_email(
        script_text=state["script_text"],
        client_name=state["client_name"],
        version=state["version"],
        deadline_hours=state.get("sla_hours", 48),
    )

    subject = (
        f"[Scrollhouse] Script Approval Request — "
        f"{state['script_id']} [v{state['version']}]"
    )

    send_email(
        to_address=state["client_email"],
        subject=subject,
        body=email_body,
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    # Update Airtable
    set_status(state["script_id"], "Pending")
    set_sent_at(state["script_id"])

    return {
        "status": "pending",
        "sent_at": now_iso,
        "follow_ups_sent": 0,
    }


def wait_for_response(state: ApprovalState) -> dict:
    """
    Passthrough node — routing is handled by the conditional edge.
    """
    return {}


def _classify_response_node(state: ApprovalState) -> dict:
    """Run Chain 2 on the raw reply and store the classification result."""
    result = classify_response(state["response_raw"])
    return {
        "status": result["decision"],
        "revision_notes": result.get("revision_notes", ""),
    }


def send_followup_1(state: ApprovalState) -> dict:
    """Send the first follow-up email (24 h cadence)."""
    subject = (
        f"[Scrollhouse] Reminder: Script {state['script_id']} "
        f"[v{state['version']}] — awaiting your feedback"
    )
    body = (
        f"Hi {state['client_name']},\n\n"
        f"Just a friendly reminder that we're waiting on your feedback for "
        f"script {state['script_id']} [v{state['version']}].\n\n"
        f"If you've already replied, please disregard this note. Otherwise, "
        f"could you take a moment to review and respond?\n\n"
        f"Thanks so much!\n— The Scrollhouse Team"
    )
    send_email(state["client_email"], subject, body)
    increment_follow_ups(state["script_id"], state.get("follow_ups_sent", 0))

    return {"follow_ups_sent": 1}


def send_followup_2(state: ApprovalState) -> dict:
    """Send the second follow-up email (48 h cadence)."""
    subject = (
        f"[Scrollhouse] Second Reminder: Script {state['script_id']} "
        f"[v{state['version']}] — feedback needed"
    )
    body = (
        f"Hi {state['client_name']},\n\n"
        f"We haven't heard back regarding script {state['script_id']} "
        f"[v{state['version']}] and wanted to check in once more.\n\n"
        f"Your feedback is important so we can keep the project moving. "
        f"Please reply at your earliest convenience.\n\n"
        f"Best,\n— The Scrollhouse Team"
    )
    send_email(state["client_email"], subject, body)
    increment_follow_ups(state["script_id"], state.get("follow_ups_sent", 0))

    return {"follow_ups_sent": 2}


def escalate(state: ApprovalState) -> dict:
    """No response after 72 h — escalate to the account manager."""
    set_status(state["script_id"], "Escalated")

    subject = (
        f"[Scrollhouse ESCALATION] No client response — "
        f"{state['script_id']} [v{state['version']}]"
    )
    body = (
        f"Hi,\n\n"
        f"Client {state['client_name']} ({state['client_email']}) has not "
        f"responded to script {state['script_id']} [v{state['version']}] "
        f"after two follow-up emails.\n\n"
        f"Please reach out to the client directly to unblock the project.\n\n"
        f"— Scrollhouse Approval Bot"
    )
    send_email(state.get("account_manager_email", ""), subject, body)

    return {"status": "escalated"}


def process_approved(state: ApprovalState) -> dict:
    """Client approved — update Airtable."""
    set_status(state["script_id"], "Approved")
    return {"status": "approved"}


def process_revision(state: ApprovalState) -> dict:
    """Client requested revisions — extract notes via Chain 3, persist."""
    notes = extract_revision_notes(
        reply_text=state["response_raw"],
        decision="revision",
    )
    save_revision_notes(state["script_id"], notes)
    set_status(state["script_id"], "Revision")

    return {"status": "revision", "revision_notes": notes}


def process_rejected(state: ApprovalState) -> dict:
    """Client rejected — update Airtable."""
    set_status(state["script_id"], "Rejected")
    return {"status": "rejected"}


def handle_call_request(state: ApprovalState) -> dict:
    """Client wants a call — pause workflow and notify account manager."""
    set_status(state["script_id"], "Paused")
    update_script_fields(state["script_id"], {"Status": "Paused"})

    subject = (
        f"[Scrollhouse] Call Requested — {state['script_id']} "
        f"[v{state['version']}]"
    )
    body = (
        f"Hi,\n\n"
        f"Client {state['client_name']} has requested a call to discuss "
        f"script {state['script_id']} [v{state['version']}].\n\n"
        f"Please schedule a call with {state['client_email']}.\n\n"
        f"— Scrollhouse Approval Bot"
    )
    send_email(state.get("account_manager_email", ""), subject, body)

    return {"status": "paused"}


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional routing helpers
# ═══════════════════════════════════════════════════════════════════════════════


def route_after_wait(state: ApprovalState) -> str:
    """Decide where to go after the wait_for_response node."""
    if state.get("response_raw"):
        return "classify_response"

    follow_ups = state.get("follow_ups_sent", 0)
    if follow_ups == 0:
        return "send_followup_1"
    elif follow_ups == 1:
        return "send_followup_2"
    else:
        return "escalate"


def route_after_classify(state: ApprovalState) -> str:
    """Route based on the classified decision stored in state['status']."""
    decision = state.get("status", "")
    mapping = {
        "approved": "process_approved",
        "revision": "process_revision",
        "rejected": "process_rejected",
        "call_requested": "handle_call_request",
    }
    return mapping.get(decision, "process_revision")


# ═══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ═══════════════════════════════════════════════════════════════════════════════


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph approval workflow."""
    graph = StateGraph(ApprovalState)

    # Register nodes
    graph.add_node("send_approval", send_approval)
    graph.add_node("wait_for_response", wait_for_response)
    graph.add_node("classify_response", _classify_response_node)
    graph.add_node("send_followup_1", send_followup_1)
    graph.add_node("send_followup_2", send_followup_2)
    graph.add_node("escalate", escalate)
    graph.add_node("process_approved", process_approved)
    graph.add_node("process_revision", process_revision)
    graph.add_node("process_rejected", process_rejected)
    graph.add_node("handle_call_request", handle_call_request)

    # Entry point
    graph.set_entry_point("send_approval")

    # Fixed edges
    graph.add_edge("send_approval", "wait_for_response")
    graph.add_edge("send_followup_1", END)
    graph.add_edge("send_followup_2", END)
    graph.add_edge("escalate", END)
    graph.add_edge("process_approved", END)
    graph.add_edge("process_revision", END)
    graph.add_edge("process_rejected", END)
    graph.add_edge("handle_call_request", END)

    # Conditional edges
    graph.add_conditional_edges(
        "wait_for_response",
        route_after_wait,
        {
            "classify_response": "classify_response",
            "send_followup_1": "send_followup_1",
            "send_followup_2": "send_followup_2",
            "escalate": "escalate",
        },
    )
    graph.add_conditional_edges(
        "classify_response",
        route_after_classify,
        {
            "process_approved": "process_approved",
            "process_revision": "process_revision",
            "process_rejected": "process_rejected",
            "handle_call_request": "handle_call_request",
        },
    )

    return graph.compile()


# Pre-compiled graph instance for import convenience
approval_graph = build_graph()

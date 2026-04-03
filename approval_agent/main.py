"""
main.py — FastAPI application exposing three endpoints for the
Scrollhouse Script Approval Agent.

Endpoints:
    POST /start-approval   — kick off the approval workflow
    POST /receive-reply    — inject a client reply and resume the workflow
    GET  /status/{script_id} — check current Airtable record
    GET  /                   — serves the dashboard UI
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from airtable_client import get_script_fields
from graph import approval_graph

app = FastAPI(
    title="Scrollhouse Script Approval Agent",
    description="Automates the content script approval loop with LangGraph + Gemini.",
    version="1.0.0",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve dashboard UI ─────────────────────────────────────────────────────
_INDEX_HTML = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return _INDEX_HTML.read_text(encoding="utf-8")


# ─── In-memory state store (keyed by script_id) ─────────────────────────────
# In production you'd persist this to Redis / a database.
_state_store: dict[str, dict] = {}


# ─── Request schemas ─────────────────────────────────────────────────────────


class StartApprovalRequest(BaseModel):
    script_id: str


class ReceiveReplyRequest(BaseModel):
    script_id: str
    reply_text: str


# ═══════════════════════════════════════════════════════════════════════════════
# POST /start-approval
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/start-approval")
def start_approval(req: StartApprovalRequest):
    """
    Fetch the script record from Airtable, build an initial ApprovalState,
    and execute the LangGraph from the send_approval entry point.
    """
    fields = get_script_fields(req.script_id)
    if not fields:
        raise HTTPException(
            status_code=404,
            detail=f"Script {req.script_id} not found in Airtable.",
        )

    # Build initial state from Airtable fields
    initial_state = {
        "script_id": fields.get("Script ID", req.script_id),
        "script_text": fields.get("Script Text", ""),
        "version": int(fields.get("Version", 1)),
        "client_name": fields.get("Client Name", ""),
        "client_email": fields.get("Client Email", ""),
        "client_channel": "email",
        "sla_hours": 48,
        "sent_at": "",
        "follow_ups_sent": int(fields.get("Follow Ups Sent", 0)),
        "status": "pending",
        "response_raw": "",
        "revision_notes": "",
        "account_manager_email": fields.get("Account Manager", ""),
    }

    # Run the graph — it will send the approval email, update Airtable,
    # then reach wait_for_response → no reply yet → follow-up or END.
    result = approval_graph.invoke(initial_state)

    # Cache the latest state for future /receive-reply calls
    _state_store[req.script_id] = result

    return {"status": "started", "script_id": req.script_id}


# ═══════════════════════════════════════════════════════════════════════════════
# POST /receive-reply
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/receive-reply")
def receive_reply(req: ReceiveReplyRequest):
    """
    Accept a client reply, inject it into the existing state,
    and re-run the graph from wait_for_response to classify + act.
    """
    # Try cached state first, then fall back to Airtable
    cached = _state_store.get(req.script_id)

    if not cached:
        fields = get_script_fields(req.script_id)
        if not fields:
            raise HTTPException(
                status_code=404,
                detail=f"Script {req.script_id} not found.",
            )
        cached = {
            "script_id": fields.get("Script ID", req.script_id),
            "script_text": fields.get("Script Text", ""),
            "version": int(fields.get("Version", 1)),
            "client_name": fields.get("Client Name", ""),
            "client_email": fields.get("Client Email", ""),
            "client_channel": "email",
            "sla_hours": 48,
            "sent_at": fields.get("Sent At", ""),
            "follow_ups_sent": int(fields.get("Follow Ups Sent", 0)),
            "status": fields.get("Status", "Pending").lower(),
            "response_raw": "",
            "revision_notes": fields.get("Revision Notes", ""),
            "account_manager_email": fields.get("Account Manager", ""),
        }

    # Inject the client reply
    cached["response_raw"] = req.reply_text

    # Build a small sub-graph run: wait_for_response → classify → process
    # We invoke the full graph but with response_raw populated so the
    # conditional edge routes straight to classify_response.
    from graph import build_graph

    reply_graph = build_graph()

    # We want to skip send_approval (already done), so we build a
    # purpose-built mini graph for the reply path.
    from langgraph.graph import END, StateGraph
    from graph import (
        wait_for_response,
        _classify_response_node,
        process_approved,
        process_revision,
        process_rejected,
        handle_call_request,
        route_after_wait,
        route_after_classify,
    )
    from state import ApprovalState

    reply_g = StateGraph(ApprovalState)
    reply_g.add_node("wait_for_response", wait_for_response)
    reply_g.add_node("classify_response", _classify_response_node)
    reply_g.add_node("process_approved", process_approved)
    reply_g.add_node("process_revision", process_revision)
    reply_g.add_node("process_rejected", process_rejected)
    reply_g.add_node("handle_call_request", handle_call_request)

    reply_g.set_entry_point("wait_for_response")

    reply_g.add_conditional_edges(
        "wait_for_response",
        route_after_wait,
        {
            "classify_response": "classify_response",
            "send_followup_1": END,   # won't be hit since response_raw is set
            "send_followup_2": END,
            "escalate": END,
        },
    )
    reply_g.add_conditional_edges(
        "classify_response",
        route_after_classify,
        {
            "process_approved": "process_approved",
            "process_revision": "process_revision",
            "process_rejected": "process_rejected",
            "handle_call_request": "handle_call_request",
        },
    )
    reply_g.add_edge("process_approved", END)
    reply_g.add_edge("process_revision", END)
    reply_g.add_edge("process_rejected", END)
    reply_g.add_edge("handle_call_request", END)

    compiled = reply_g.compile()
    result = compiled.invoke(cached)

    _state_store[req.script_id] = result

    return {
        "status": "processed",
        "decision": result.get("status", "unknown"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GET /status/{script_id}
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/status/{script_id}")
def get_status(script_id: str):
    """Return the full Airtable record for a given Script ID."""
    fields = get_script_fields(script_id)
    if not fields:
        raise HTTPException(
            status_code=404,
            detail=f"Script {script_id} not found in Airtable.",
        )
    return fields

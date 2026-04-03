"""
chains.py — Three LangChain chains powered by Gemini 1.5 Flash.

Chain 1: Approval message formatter
Chain 2: Response classifier
Chain 3: Revision note extractor
"""

import json
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
load_dotenv()


# ─── Lazy LLM initialisation ────────────────────────────────────────────────
# Deferred so the module can be imported even when GOOGLE_API_KEY is not yet
# set.  The key is only required when a chain is actually invoked.

@lru_cache(maxsize=1)
def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.  "
            "Please add it to your .env file before invoking any chain."
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Chain 1 — Approval Message Formatter
# ═══════════════════════════════════════════════════════════════════════════════

_approval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a professional content agency assistant at Scrollhouse. "
                "Write a clear, friendly, and professional plain-text email body "
                "requesting script approval from a client.\n\n"
                "Rules:\n"
                "- Address the client by first name.\n"
                "- Clearly state the script version as [v{{version}}].\n"
                "- Include the full script text so the client can review inline.\n"
                "- End with a clear call to action: ask the client to reply with "
                "  approval or specific feedback.\n"
                "- Mention the deadline ({{deadline_hours}} hours from now).\n"
                "- Keep tone warm but concise — no fluff.\n"
                "- Output ONLY the email body (no subject line, no headers)."
            ),
        ),
        (
            "human",
            (
                "Client name: {client_name}\n"
                "Script version: v{version}\n"
                "Deadline: {deadline_hours} hours\n\n"
                "--- SCRIPT ---\n{script_text}\n--- END SCRIPT ---"
            ),
        ),
    ]
)


def _build_approval_chain():
    return _approval_prompt | _get_llm() | StrOutputParser()


def format_approval_email(
    script_text: str,
    client_name: str,
    version: int,
    deadline_hours: int = 48,
) -> str:
    """Run Chain 1 and return the formatted email body."""
    return _build_approval_chain().invoke(
        {
            "script_text": script_text,
            "client_name": client_name,
            "version": version,
            "deadline_hours": deadline_hours,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Chain 2 — Response Classifier
# ═══════════════════════════════════════════════════════════════════════════════

_classifier_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a response classifier for a content approval workflow.\n\n"
                "Given a client's reply to a script approval request, classify it "
                "into exactly ONE of these categories:\n"
                "  - approved\n"
                "  - revision\n"
                "  - rejected\n"
                "  - call_requested\n\n"
                "Also extract any revision notes (empty string if none) and rate "
                "your confidence from 0.0 to 1.0.\n\n"
                "FEW-SHOT EXAMPLES:\n"
                'Reply: "looks good" → {{"decision":"approved","revision_notes":"","confidence":0.95}}\n'
                'Reply: "yeah that works 👍" → {{"decision":"approved","revision_notes":"","confidence":0.92}}\n'
                'Reply: "can we tweak the opening?" → {{"decision":"revision","revision_notes":"Tweak the opening section","confidence":0.90}}\n'
                'Reply: "please change the intro and make it shorter" → {{"decision":"revision","revision_notes":"Change the intro and make it shorter","confidence":0.93}}\n'
                'Reply: "this doesn\'t work for us" → {{"decision":"rejected","revision_notes":"","confidence":0.88}}\n'
                'Reply: "can we jump on a call?" → {{"decision":"call_requested","revision_notes":"","confidence":0.94}}\n'
                'Reply: "looks mostly fine, just fix the last line" → {{"decision":"revision","revision_notes":"Fix the last line","confidence":0.91}}\n\n'
                "OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:\n"
                '{{"decision":"...","revision_notes":"...","confidence":0.XX}}'
            ),
        ),
        ("human", "Client reply:\n{reply_text}"),
    ]
)


def _build_classifier_chain():
    return _classifier_prompt | _get_llm() | StrOutputParser()


def classify_response(reply_text: str) -> dict[str, Any]:
    """
    Run Chain 2 and return structured classification.

    Returns:
        {"decision": str, "revision_notes": str, "confidence": float}
    """
    raw = _build_classifier_chain().invoke({"reply_text": reply_text})
    # Strip markdown fences if the model wraps them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback — treat as revision to be safe
        result = {
            "decision": "revision",
            "revision_notes": reply_text,
            "confidence": 0.5,
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Chain 3 — Revision Note Extractor
# ═══════════════════════════════════════════════════════════════════════════════

_extractor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a revision note extractor for a content agency.\n\n"
                "Given the client's raw reply and the classifier's decision, "
                "produce a structured task description that a scriptwriter can "
                "act on immediately.\n\n"
                "Format your output as plain text with:\n"
                "1. APPROVED SECTIONS — list which parts are fine.\n"
                "2. CHANGES REQUESTED — bullet-point each change with specific "
                "   instructions.\n"
                "3. PRIORITY — High / Medium / Low.\n\n"
                "Be concise and actionable. Do not repeat the original script."
            ),
        ),
        (
            "human",
            (
                "Client reply: {reply_text}\n"
                "Classified decision: {decision}\n"
            ),
        ),
    ]
)


def _build_extractor_chain():
    return _extractor_prompt | _get_llm() | StrOutputParser()


def extract_revision_notes(reply_text: str, decision: str) -> str:
    """Run Chain 3 and return structured revision notes."""
    return _build_extractor_chain().invoke(
        {"reply_text": reply_text, "decision": decision}
    )

"""
app/frontend/ui.py
──────────────────────────────────────────────
WHAT THIS FILE DOES:
  This is the Gradio chat interface that lawyers use.
  It sends questions to FastAPI and displays answers
  with a citations panel.

FEATURES:
  - Chat window with multi-turn conversation
  - Act filter dropdown (AUTO/BNS/BNSS/BSA/DPDP/ALL)
  - Citations panel showing exact sections used
  - New session button to start fresh
  - Example questions to get started quickly
  - Shows response time and cache status

HOW TO RUN:
  python -m app.frontend.ui
  Then open: http://localhost:7860
"""
from __future__ import annotations

import httpx
import gradio as gr

from app.core.config import get_settings

settings = get_settings()
API_URL  = settings.gradio_api_url


# ── API helper ────────────────────────────────────────────────────────

async def _ask_api(
    question:   str,
    act_filter: str,
    session_id: str | None,
) -> dict:
    """
    Sends question to FastAPI /query endpoint.
    Returns the full response dict.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{API_URL}/query",
            json={
                "question":   question,
                "act_filter": None if act_filter == "AUTO" else act_filter,
                "session_id": session_id or None,
            },
        )
        response.raise_for_status()
        return response.json()


async def _new_session_api() -> str:
    """Creates a new session via FastAPI and returns the session_id."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(f"{API_URL}/sessions", json={})
        response.raise_for_status()
        return response.json()["session_id"]


# ── Citation formatter ────────────────────────────────────────────────

def _format_citations(citations: list[dict]) -> str:
    """
    Formats the citations list into readable markdown
    for display in the citations panel.
    """
    if not citations:
        return "*No citations retrieved.*"

    lines = ["### 📚 Sections Referenced\n"]

    for i, c in enumerate(citations, 1):
        lines.append(
            f"**{i}. {c['act_full_name']}**\n"
            f"- Section: `{c['section_number']}`\n"
            f"- Title: {c['title']}\n"
            f"- Pages: {c['pages']}\n"
            f"- Relevance score: `{c['score']}`\n"
            f"- *{c['summary']}*\n"
        )

    return "\n".join(lines)


# ── Main chat function ────────────────────────────────────────────────

async def chat(
    message:       str,
    history:       list,
    act_filter:    str,
    session_state: dict,
):
    """
    Called every time the lawyer submits a question.

    Flow:
      1. Add question to chat history immediately
      2. Send to FastAPI
      3. Update chat with answer
      4. Update citations panel
    """
    if not message.strip():
        yield history, "*Ask a question to see citations.*", session_state
        return

    session_id = session_state.get("session_id")

    # Show question immediately in chat
    history = history + [[message, None]]
    yield history, "*Searching legal sections...*", session_state

    try:
        data = await _ask_api(message, act_filter, session_id)
    except httpx.HTTPStatusError as e:
        error_msg = f"⚠️ API error {e.response.status_code}: {e.response.text}"
        history[-1][1] = error_msg
        yield history, "*Error occurred.*", session_state
        return
    except httpx.ConnectError:
        error_msg = (
            "⚠️ Cannot connect to API server.\n\n"
            "Make sure FastAPI is running:\n"
            "`uvicorn app.api.main:app --reload --port 8000`"
        )
        history[-1][1] = error_msg
        yield history, "*Connection error.*", session_state
        return
    except Exception as e:
        history[-1][1] = f"⚠️ Unexpected error: {str(e)}"
        yield history, "*Error occurred.*", session_state
        return

    # Update session_id if this was a new session
    if not session_id:
        session_state = {**session_state, "session_id": data["session_id"]}

    # Build answer with metadata footer
    cache_badge = "⚡ cached" if data["cached"] else f"🔍 {data['latency_ms']}ms"
    acts_badge  = " · ".join(data["act_scope"])
    footer      = f"\n\n---\n*{cache_badge} | Acts searched: {acts_badge}*"

    history[-1][1] = data["answer"] + footer

    # Format citations for the panel
    citations_md = _format_citations(data["citations"])

    yield history, citations_md, session_state


async def start_new_session(session_state: dict):
    """Clears chat and starts a fresh session."""
    sid = await _new_session_api()
    return [], "*New session started. Ask your first question.*", {"session_id": sid}


# ── Build the UI ──────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """
    Creates and returns the Gradio UI.
    Called once when the app starts.
    """
    with gr.Blocks(
        title="Legal RAG — Indian Laws",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
        ),
    ) as demo:

        # Session state — stores session_id across messages
        session_state = gr.State({"session_id": None})

        # ── Header ────────────────────────────────────────────────────
        gr.Markdown("""
        # ⚖️ Legal RAG Chatbot — Indian Laws
        **BNS 2023** (replaces IPC) · **BNSS 2023** (replaces CrPC) ·
        **BSA 2023** (replaces Evidence Act) · **DPDP Act 2023**

        Ask any legal question and get answers with **exact section citations**.
        """)

        # ── Main layout ───────────────────────────────────────────────
        with gr.Row():

            # Left column — chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Legal Q&A",
                    height=500,
                    show_copy_button=True,
                    render_markdown=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g. What is the punishment for murder under BNS 2023?",
                        label="Your question",
                        lines=2,
                        scale=5,
                    )
                    act_filter = gr.Dropdown(
                        choices=["AUTO", "BNS", "BNSS", "BSA", "DPDP", "ALL"],
                        value="AUTO",
                        label="Act filter",
                        scale=1,
                    )

                with gr.Row():
                    submit_btn     = gr.Button("Ask ⚖️", variant="primary", scale=4)
                    new_session_btn = gr.Button("🔄 New session", scale=1)

            # Right column — citations
            with gr.Column(scale=2):
                citation_box = gr.Markdown(
                    value="*Ask a question to see section citations here.*",
                    label="Citations",
                )

        # ── Example questions ─────────────────────────────────────────
        gr.Examples(
            examples=[
                ["What is the punishment for murder under BNS 2023?",           "BNS"],
                ["What are the bail provisions for non-bailable offences?",      "BNSS"],
                ["How is electronic evidence admissible under BSA 2023?",        "BSA"],
                ["What are the rights of a data principal under DPDP Act?",      "DPDP"],
                ["What changed from IPC Section 302 to BNS?",                   "AUTO"],
                ["What is the FIR filing procedure under BNSS?",                 "BNSS"],
                ["Explain consent requirements under DPDP Act 2023.",            "DPDP"],
                ["What is culpable homicide under BNS?",                         "BNS"],
            ],
            inputs=[msg_input, act_filter],
            label="Example questions — click any to try",
        )

        # ── Wire up events ────────────────────────────────────────────
        submit_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, act_filter, session_state],
            outputs=[chatbot, citation_box, session_state],
        )
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, act_filter, session_state],
            outputs=[chatbot, citation_box, session_state],
        )
        new_session_btn.click(
            fn=start_new_session,
            inputs=[session_state],
            outputs=[chatbot, citation_box, session_state],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        show_api=False,
    )
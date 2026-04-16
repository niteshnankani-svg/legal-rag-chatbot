"""
app/frontend/ui.py — Gradio chat interface
HuggingFace-compatible. Queue disabled to fix stop_event bug.
"""
from __future__ import annotations
import httpx
import gradio as gr
from app.core.config import get_settings

settings = get_settings()
API_URL = settings.gradio_api_url


def _ask_api(question: str, act_filter: str, session_id: str) -> dict:
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{API_URL}/query", json={
            "question":   question,
            "act_filter": None if act_filter == "AUTO" else act_filter,
            "session_id": session_id or None,
        })
        r.raise_for_status()
        return r.json()


def _new_session() -> str:
    try:
        with httpx.Client(timeout=15) as client:
            r = client.post(f"{API_URL}/sessions", json={})
            r.raise_for_status()
            return r.json()["session_id"]
    except Exception:
        return ""


def _check_backend() -> bool:
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{API_URL}/health")
            return r.status_code == 200
    except Exception:
        return False


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return "*No citations retrieved.*"
    lines = ["### 📚 Sections Referenced\n"]
    for i, c in enumerate(citations, 1):
        lines.append(
            f"**{i}. {c['act_full_name']}**\n"
            f"- Section: `{c['section_number']}`\n"
            f"- Title: {c['title']}\n"
            f"- Pages: {c['pages']}\n"
            f"- Relevance: `{c['score']}`\n"
            f"- *{c['summary']}*\n"
        )
    return "\n".join(lines)


def chat(message: str, history: list, act_filter: str, session_id: str):
    if not message.strip():
        return history, "*Ask a question to see citations.*", session_id

    if not _check_backend():
        history = history + [[message,
            "⏳ **Backend warming up.** Please wait 30 seconds and try again."
        ]]
        return history, "*Backend warming up...*", session_id

    history = history + [[message, None]]

    try:
        data = _ask_api(message, act_filter, session_id or None)
    except httpx.ConnectError:
        history[-1][1] = "⏳ **Backend starting up.** Please wait 30 seconds and try again."
        return history, "*Connecting...*", session_id
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            history[-1][1] = "⚠️ **Index loading.** Please wait 30 seconds and try again."
        else:
            history[-1][1] = f"⚠️ Error {e.response.status_code}"
        return history, "*Error.*", session_id
    except Exception as e:
        history[-1][1] = f"⚠️ Error: {str(e)}"
        return history, "*Error.*", session_id

    new_session_id = data.get("session_id", session_id)
    cache_badge = "⚡ cached" if data["cached"] else f"🔍 {data['latency_ms']}ms"
    acts_badge = " · ".join(data["act_scope"])
    history[-1][1] = data["answer"] + f"\n\n---\n*{cache_badge} | Acts: {acts_badge}*"
    return history, _format_citations(data["citations"]), new_session_id


def start_new_session(session_id: str):
    sid = _new_session()
    return [], "*New session started.*", sid


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Legal RAG — Indian Laws",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    ) as demo:

        session_id = gr.State("")

        gr.Markdown("""
        # ⚖️ Legal RAG Chatbot — Indian Laws
        **BNS 2023** · **BNSS 2023** · **BSA 2023** · **DPDP Act 2023**

        Ask any legal question and get answers with **exact section citations**.
        *First response may take 30–60 seconds on startup.*
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Legal Q&A", height=500,
                    show_copy_button=True, render_markdown=True,
                    bubble_full_width=False,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g. What is the punishment for murder under BNS 2023?",
                        label="Your question", lines=2, scale=5,
                    )
                    act_filter = gr.Dropdown(
                        choices=["AUTO", "BNS", "BNSS", "BSA", "DPDP", "ALL"],
                        value="AUTO", label="Act filter", scale=1,
                    )
                with gr.Row():
                    submit_btn      = gr.Button("Ask ⚖️", variant="primary", scale=4)
                    new_session_btn = gr.Button("🔄 New session", scale=1)

            with gr.Column(scale=2):
                citation_box = gr.Markdown(
                    value="*Ask a question to see section citations here.*",
                    label="Citations",
                )

        gr.Examples(
            examples=[
                ["What is the punishment for murder under BNS 2023?",      "BNS"],
                ["What are the bail provisions for non-bailable offences?", "BNSS"],
                ["How is electronic evidence admissible under BSA 2023?",   "BSA"],
                ["What are the rights of a data principal under DPDP Act?", "DPDP"],
                ["What changed from IPC Section 302 to BNS?",              "AUTO"],
                ["What is the FIR filing procedure under BNSS?",            "BNSS"],
                ["Explain consent requirements under DPDP Act 2023.",       "DPDP"],
                ["What is culpable homicide under BNS?",                    "BNS"],
            ],
            inputs=[msg_input, act_filter],
            label="Example questions — click any to try",
        )

        # Wire events — no queue
        submit_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, act_filter, session_id],
            outputs=[chatbot, citation_box, session_id],
            queue=False,
        )
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, act_filter, session_id],
            outputs=[chatbot, citation_box, session_id],
            queue=False,
        )
        new_session_btn.click(
            fn=start_new_session,
            inputs=[session_id],
            outputs=[chatbot, citation_box, session_id],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        show_api=False,
    )

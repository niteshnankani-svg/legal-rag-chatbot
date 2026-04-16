"""
app.py — HuggingFace Spaces entry point
Downloads index from HuggingFace Dataset repo at startup.
Then runs FastAPI + Gradio together in one process.
"""
import threading
import uvicorn
import os
import sys
import time
import httpx
from pathlib import Path

os.environ["GRADIO_API_URL"] = "http://127.0.0.1:8000"
os.environ["SQLITE_DB_PATH"] = "/tmp/legal_chat.db"
os.environ["INDEX_DIR"]      = "./data/index"
os.environ["REDIS_HOST"]     = "localhost"

INDEX_FILES = [
    "BNS_embeddings.pkl",  "BNS_index.json",  "BNS_pages.json",
    "BNSS_embeddings.pkl", "BNSS_index.json", "BNSS_pages.json",
    "BSA_embeddings.pkl",  "BSA_index.json",  "BSA_pages.json",
    "DPDP_embeddings.pkl", "DPDP_index.json", "DPDP_pages.json",
]

def download_index():
    target_dir = Path("data/index")
    target_dir.mkdir(parents=True, exist_ok=True)
    if (target_dir / "BNS_embeddings.pkl").exists():
        print("Index already exists — skipping download")
        return
    print("Downloading index files from HuggingFace Dataset...")
    from huggingface_hub import hf_hub_download
    for filename in INDEX_FILES:
        print(f"  Downloading {filename}...")
        hf_hub_download(
            repo_id="nitz0219/legal-rag-index-v1",
            repo_type="dataset",
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  Done: {filename}")
    print("All index files downloaded!")

download_index()

def run_fastapi():
    try:
        uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"FastAPI error: {e}", file=sys.stderr)

print("Starting FastAPI backend...")
fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
fastapi_thread.start()

def wait_for_fastapi(timeout=120):
    print("Waiting for FastAPI to be ready...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get("http://127.0.0.1:8000/health", timeout=3)
            if r.status_code == 200:
                print("FastAPI is ready!")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("WARNING: FastAPI did not start in time.")
    return False

wait_for_fastapi(timeout=120)

print("Starting Gradio UI...")
from app.frontend.ui import build_ui
demo = build_ui()
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

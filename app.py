# app.py — HuggingFace Spaces entry point
from app.frontend.ui import build_ui

demo = build_ui()
demo.launch()
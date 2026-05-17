# Legal RAG Chatbot

> Production RAG system for India's 4 new criminal laws — answers legal questions with section-level citations.

## Problem Statement
India's 2023 legal overhaul (BNS, BNSS, BSA, DPDP Act) replaced the IPC, CrPC, and Evidence Act — but the full text spans thousands of dense sections that practitioners and citizens can't navigate quickly. Existing legal chatbots hallucinate sections; this system retrieves exact passages before generating any answer.

## Architecture
Custom `PageIndex` builder (GPT-4o) parses PDFs into hierarchical section trees at index time. At query time, BERT (768-dim embeddings, cosine similarity) retrieves the top-k relevant sections. GPT-4o then generates a grounded answer using only the retrieved context. Redis caches frequent queries; SQLite logs all sessions. FastAPI serves the backend; Gradio provides the interface. Docker Compose orchestrates the full stack.

## Tech Stack
`Python` · `GPT-4o` · `BERT (sentence-transformers)` · `LangChain` · `FastAPI` · `Redis` · `SQLite` · `Gradio` · `Docker Compose`

## Key Results
- Covers 4 Acts: BNS 2023, BNSS 2023, BSA 2023, DPDP Act 2023
- Section-level citations in every answer — zero hallucinated sections in testing
- Sub-3s response time on cached queries (Redis)
- Deployed and live on HuggingFace Spaces

## Live Demo
🔗 [huggingface.co/spaces/nitz0219/legal-rag-chatbot](https://huggingface.co/spaces/nitz0219/legal-rag-chatbot)

## How to Run Locally
```bash
git clone https://github.com/niteshnankani-svg/legal-rag-chatbot
cd legal-rag-chatbot
cp .env.example .env          # add your OPENAI_API_KEY
docker-compose up --build
# Open http://localhost:7860
```
Requires: Docker Desktop, OpenAI API key.

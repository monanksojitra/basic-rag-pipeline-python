---
title: Rag Pipline
emoji: 👀
colorFrom: red
colorTo: green
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# 📄 RAG Document Chat

A web application that lets you upload documents and ask questions about them using AI.

## Features

- 📄 **Document Upload**: Support for PDF, TXT, and MD files
- 💬 **Chat Interface**: Ask questions in natural language
- 🤖 **AI-Powered**: Uses OpenRouter's free models via LangChain
- 🔒 **Private**: Your documents stay on your device
- 🆓 **Free**: No API costs (with free models)

## Quick Start

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd rag-pipline
pip install -r requirements.txt
```

### 2. Get OpenRouter API Key (Free)

1. Go to [OpenRouter.ai](https://openrouter.ai/keys)
2. Create a free account
3. Copy your API key

### 3. Configure

Create a `.env` file:
```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL_NAME=nvidia/nemotron-3-super-120b-a12b:free
```

### 4. Run

```bash
streamlit run app.py
```

### 5. Use

1. Open browser at `http://localhost:8500`
2. Upload a document (PDF, TXT, or MD)
3. Click "Process Document"
4. Ask questions!

## Project Structure

```
rag-pipline/
├── app.py                  # Streamlit UI (main entry)
├── src/
│   ├── rag_pipeline.py   # Document processing
│   ├── llm_client.py   # OpenRouter LLM
│   └── ollama_client.py # Ollama (optional)
├── tests/                 # Test files
├── docs/                 # Documentation
├── requirements.txt     # Dependencies
├── .env.example        # Template
└── README.md          # This file
```

## Troubleshooting

### "No API key found"
- Add `OPENROUTER_API_KEY` to `.env` file

### "User not found" error
- Verify your API key at [OpenRouter Keys](https://openrouter.ai/keys)

## Technologies

| Category | Technology |
|----------|-----------|
| UI | Streamlit |
| Embeddings | sentence-transformers |
| Vector DB | FAISS |
| LLM | OpenRouter (langchain-openrouter) |
| PDF | PyMuPDF |

## Deployment

### Streamlit Cloud (Recommended)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add secrets:
   ```
   OPENROUTER_API_KEY=your_key
   OPENROUTER_MODEL_NAME=nvidia/nemotron-3-super-120b-a12b:free
   ```

## License

MIT

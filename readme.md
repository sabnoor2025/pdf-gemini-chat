# 📄 Chat with Your PDF (Gemini + Gradio + FAISS)

An interactive **PDF Question Answering App** powered by **Google Gemini AI**, **FAISS embeddings**, and a clean **Gradio UI**.  
Upload any PDF, and chat with it like a personal assistant. ⚡

---

## 🚀 Features
- 📑 **Upload PDFs** and extract text automatically.  
- 🔍 **Semantic search with FAISS** for efficient retrieval.  
- 🤖 **Context-aware answers** generated with Gemini AI.  
- 🎨 **Interactive Gradio interface** for smooth chatting.  
- 🔑 **Environment variable support** with `.env` for API key management.  

---

## 🛠️ Tech Stack
- [Python 3.11](https://www.python.org/downloads/release/python-3119/)  
- [Gradio](https://gradio.app/) – UI Framework  
- [Google Gemini (`google-genai`)](https://ai.google.dev/gemini-api/docs/libraries) – LLM API  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector Search  
- [NumPy](https://numpy.org/) – Embedding handling  
- [PyPDF2](https://pypi.org/project/pypdf2/) – PDF text extraction  
- [python-dotenv](https://pypi.org/project/python-dotenv/) – Env variable loader  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pdf-gemini-chat.git
cd pdf-gemini-chat

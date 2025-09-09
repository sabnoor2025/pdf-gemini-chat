import os
import gradio as gr
import PyPDF2
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- API Setup ---
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# --- Globals ---
index = None
chunks = []

# --- PDF functions ---
def load_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1400, overlap=200):
    words = text.split()
    if len(words) <= chunk_size:
        yield text
        return
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        yield chunk
        start += chunk_size - overlap

def build_index(chunks, batch_size=10):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        for ch in batch:
            try:
                resp = client.models.embed_content(
                    model="embedding-001",
                    contents=[ch]
                )
                emb = resp.embeddings[0].values
                embeddings.append(emb)
            except Exception as e:
                print(f"Embedding failed for chunk {i}: {e}")
                embeddings.append(np.zeros(768))  # fallback dim
    embeddings = np.array(embeddings, dtype="float32")
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)
    return idx

def retrieve(query, index, chunks, k=3):
    try:
        resp = client.models.embed_content(
            model="embedding-001",
            contents=[query]
        )
        q_emb = np.array([resp.embeddings[0].values], dtype="float32")
    except Exception as e:
        print(f"Failed to get embedding: {e}")
        return []
    scores, ids = index.search(q_emb, k)
    return [(chunks[i], scores[0][j]) for j, i in enumerate(ids[0])]

def answer_query(query, index, chunks, k=3):
    top_chunks_with_scores = retrieve(query, index, chunks, k=k)
    context = "\n\n".join([chunk for chunk, score in top_chunks_with_scores])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer clearly:"
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[prompt]
        )
        return resp.text
    except Exception as e:
        return f"Failed to generate answer: {e}"

# --- PDF processing ---
def process_pdf(pdf_file):
    global index, chunks
    pdf_path = pdf_file.name
    raw_text = load_pdf(pdf_path)
    chunks = list(chunk_text(raw_text))
    index = build_index(chunks)
    # Hide uploader, show chat, start chatbot with a welcome msg
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        [("", "📑 PDF loaded! You can now ask me questions.")]
    )

# --- Chat function ---
def chat_with_pdf(message, chat_history):
    global index, chunks
    response = answer_query(message, index, chunks)
    chat_history.append((message, response))
    return "", chat_history

# --- Gradio UI ---
with gr.Blocks(title="Chat with your PDF") as demo:
    gr.Markdown("## 📄 Chat with your PDF")

    # Upload section
    with gr.Row(visible=True) as upload_section:
        pdf_input = gr.File(file_types=[".pdf"], label="Upload your PDF")

    # Chat section (hidden first)
    with gr.Column(visible=False) as chat_section:
        chatbot = gr.Chatbot(label="Conversation")
        user_input = gr.Textbox(
            label="Your question",
            placeholder="Ask something about the PDF...",
        )
        user_input.submit(chat_with_pdf, [user_input, chatbot], [user_input, chatbot])

    # Hook up file upload
    pdf_input.change(
        process_pdf,
        inputs=pdf_input,
        outputs=[upload_section, chat_section, chatbot],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True)

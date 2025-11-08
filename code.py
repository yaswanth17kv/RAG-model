import io
import os
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

# --- init client from OPENAI_API_KEY env var ---
client = OpenAI()  # will use OPENAI_API_KEY from environment

st.set_page_config(page_title="Simple PDF RAG", layout="centered")
st.title("📄 Simple PDF RAG Chatbot")

st.markdown("Upload a text PDF, ask a question — the app will use a small RAG pipeline to answer.")

# --- Helpers ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)

def split_words(text: str, chunk_words=400, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_words])
        chunks.append(chunk)
        i += chunk_words - overlap
    return chunks

def embed_batch(texts, model="text-embedding-3-small", batch_size=32):
    """Returns normalized numpy array (N, dim) float32."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_embs = [item.embedding for item in resp.data]
        all_embs.extend(batch_embs)
    arr = np.array(all_embs, dtype="float32")
    # normalize rows -> cosine similarity with inner product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

@st.cache_resource
def build_index(chunks):
    emb = embed_batch(chunks)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb

def retrieve(query, chunks, index, k=3):
    q = client.embeddings.create(model="text-embedding-3-small", input=query)
    q_emb = np.array(q.data[0].embedding, dtype="float32")
    q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-12)
    D, I = index.search(q_emb.reshape(1, -1), k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        results.append({"index": int(idx), "score": float(score), "chunk": chunks[idx]})
    return results

def ask_with_context(query, retrieved, gen_model="gpt-4o-mini", temperature=0.0):
    # Build a short prompt that includes chunk ids
    context = "\n\n---\n\n".join([f"[chunk {r['index']}]\n{r['chunk']}" for r in retrieved])
    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer the question. "
        "If the answer is not in context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    )
    resp = client.responses.create(
        model=gen_model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        temperature=float(temperature),
        max_output_tokens=512,
    )
    out = getattr(resp, "output_text", None)
    if out:
        return out.strip()
    # fallback parsing
    try:
        texts = []
        for block in resp.output or []:
            for c in block.get("content", []):
                if c.get("type") == "output_text":
                    texts.append(c.get("text", ""))
        return "\n".join(texts).strip()
    except Exception:
        return ""

# --- UI ---
uploaded = st.file_uploader("Upload a text PDF", type=["pdf"])
question = st.text_input("Ask a question about the PDF")
top_k = st.slider("How many chunks to retrieve (k)", 1, 6, 3)

if uploaded:
    pdf_bytes = uploaded.read()
    text = extract_text_from_pdf_bytes(pdf_bytes)
    if not text.strip():
        st.error("No text found in PDF. It might be scanned — OCR required.")
    else:
        st.info("PDF text extracted. Building index (this will call embeddings once)...")
        chunks = split_words(text, chunk_words=400, overlap=80)
        index, emb = build_index(chunks)
        st.success(f"Created {len(chunks)} chunks and FAISS index.")

        if question:
            with st.spinner("Retrieving relevant chunks..."):
                retrieved = retrieve(question, chunks, index, k=top_k)
            if not retrieved:
                st.warning("No relevant chunks found.")
            else:
                st.write("**Top chunks used:**")
                for r in retrieved:
                    st.write(f"- chunk {r['index']} (score {r['score']:.3f})")
                    st.write(r['chunk'][:600] + ("..." if len(r['chunk'])>600 else ""))
                with st.spinner("Generating answer..."):
                    answer = ask_with_context(question, retrieved)
                st.markdown("### Answer")
                st.write(answer)
else:
    st.info("Upload a PDF to begin. Make sure OPENAI_API_KEY is set in your environment.")

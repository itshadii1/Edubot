import os
from dotenv import load_dotenv
import chromadb
from langgraph.graph import StateGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from openai import OpenAI

load_dotenv()

# --- 1) Clients ---
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="")
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                           api_key=os.getenv("OPENROUTER_API_KEY"))

# --- 2) Build Chroma KB from PDF (dim=768) ---
loader = PyPDFLoader("cpp_book.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pdf_docs = splitter.split_documents(pages)
for d in pdf_docs:
    d.metadata.setdefault("source", "cpp_book.pdf")

client = chromadb.Client()
try:
    client.delete_collection("tutor-kb")
except Exception:
    pass
collection = client.create_collection("tutor-kb")

embedder = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
for i, doc in enumerate(pdf_docs):
    vec = embedder.embed_query(doc.page_content)
    collection.add(
        ids=[f"pdf-{i}"],
        documents=[doc.page_content],
        metadatas=[doc.metadata],
        embeddings=[vec]
    )

ROLE_SYSTEM_PROMPTS = {
    "beginner": "You are a friendly C++ tutor. Explain like I'm 12, use analogies.",
    "intermediate": "You are a knowledgeable C++ instructor. Be concise with examples.",
    "expert": "You are a senior C++ professor. Use technical depth and references."
}

def classify_question(question: str) -> str:
    sys_prompt = (
        "You are an assistant that classifies C++ questions into exactly one of three "
        "difficulty levels: beginner, intermediate, or expert. "
        "Reply with only the single word: beginner, intermediate, or expert."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]
    resp = openrouter_client.chat.completions.create(
        model="mistralai/mistral-7b-instruct:free",
        messages=messages
    )
    role = resp.choices[0].message.content.strip().lower()
    if role not in ROLE_SYSTEM_PROMPTS:
        role = "intermediate"
    return role

def retrieve_context(question: str) -> str:
    qvec = embedder.embed_query(question)
    res = collection.query(query_embeddings=[qvec], n_results=2, include=["documents"])
    return "\n\n".join(res["documents"][0]) if res and res.get("documents") else ""

def answer_question(role: str, context: str, question: str) -> str:
    sys_p = ROLE_SYSTEM_PROMPTS[role]
    msgs = [
        {"role": "system", "content": sys_p + "\nUse the context below:"},
        {"role": "user",   "content": f"Context:\n{context}\n\nQ: {question}"}
    ]
    # Route by role
    if role == "beginner":
        client_ = ollama_client
        model_ = "phi3:mini"
    elif role == "expert":
        client_ = openrouter_client
        model_ = "deepseek/deepseek-chat-v3-0324:free"
    else:
        client_ = openrouter_client
        model_ = "mistralai/mistral-7b-instruct:free"

    resp = client_.chat.completions.create(model=model_, messages=msgs)
    return resp.choices[0].message.content

def run_agent_once(question: str) -> dict:
    role = classify_question(question)
    ctx = retrieve_context(question)
    ans = answer_question(role, ctx, question)
    return {"role": role, "context": ctx, "answer": ans}

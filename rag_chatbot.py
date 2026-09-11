import os
import re

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 5  # sentences per chunk
CHUNK_OVERLAP = 2  # overlapping sentences between consecutive chunks
TOP_K = 3
SYSTEM_PROMPT = (
    "Answer ONLY using this context about Prashanth G. "
    "If the answer is not in the context, say you don't have that information."
)
WELCOME_MESSAGE = (
    "👋 Hi! I'm here to answer questions about Prashanth G — his background, "
    "skills, projects, and hobbies. What would you like to know?"
)

st.set_page_config(page_title="Ask Me Anything About Prashanth G", page_icon="🤖")

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef1f8 100%);
        }

        h1 {
            font-weight: 700;
            letter-spacing: -0.02em;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(49, 51, 63, 0.1);
            margin-bottom: 1.5rem;
        }

        [data-testid="stChatMessage"] {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        }

        [data-testid="stChatInput"] {
            border-radius: 12px;
        }

        details {
            background-color: #fafbfc;
            border-radius: 10px;
            padding: 0.4rem 0.75rem;
            border: 1px solid rgba(49, 51, 63, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Ask Me Anything About Prashanth G")

ABOUT_ME_PATH = os.path.join(os.path.dirname(__file__), "about_me.txt")

if not OPENAI_API_KEY:
    st.error(
        "⚠️ **OPENAI_API_KEY is missing.** Add it to a `.env` file in the project "
        "root (e.g. `OPENAI_API_KEY=sk-...`) and restart the app."
    )
    st.stop()

if not os.path.exists(ABOUT_ME_PATH):
    st.error(f"⚠️ **about_me.txt not found** at `{ABOUT_ME_PATH}`. Please add it.")
    st.stop()


def load_and_chunk(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start:start + CHUNK_SIZE])
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(sentences):
            break
    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    return [item.embedding for item in response.data]


def embed_query(question: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    return response.data[0].embedding


def top_k_chunks(question_embedding: list[float], k: int = TOP_K) -> list[str]:
    query_vec = np.array(question_embedding)
    chunk_matrix = np.array(st.session_state.embeddings)

    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_matrix / np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    similarities = chunk_norms @ query_norm

    top_indices = np.argsort(similarities)[::-1][:k]
    return [st.session_state.chunks[i] for i in top_indices]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content


if "chunks" not in st.session_state or "embeddings" not in st.session_state:
    st.session_state.chunks = load_and_chunk(ABOUT_ME_PATH)
    st.session_state.embeddings = embed_chunks(st.session_state.chunks)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE, "sources": []}
    ]

EXAMPLE_QUESTIONS = [
    "What are your hobbies?",
    "What technologies do you know?",
    "Tell me about your projects.",
]

with st.sidebar:
    st.subheader("Try an example")
    for question in EXAMPLE_QUESTIONS:
        if st.button(question, use_container_width=True):
            st.session_state.pending_question = question

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE, "sources": []}
        ]
        st.session_state.pop("pending_question", None)
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, chunk in enumerate(message["sources"], start=1):
                    st.markdown(f"**Chunk {i}:** {chunk}")

user_input = st.chat_input("Ask me anything...")
if not user_input and "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            question_embedding = embed_query(user_input)
            context_chunks = top_k_chunks(question_embedding)
            response = generate_answer(user_input, context_chunks)

        st.markdown(response)
        with st.expander("Sources"):
            for i, chunk in enumerate(context_chunks, start=1):
                st.markdown(f"**Chunk {i}:** {chunk}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": context_chunks}
    )

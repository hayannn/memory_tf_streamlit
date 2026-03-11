import os
import json
import pickle
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import streamlit as st
from pymilvus import MilvusClient, DataType
from openai import OpenAI


# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------

DEFAULT_BENE_PKL_PATH = "./data/kpfis_bene_embed_merged_200.pkl"
DEFAULT_OP_PKL_PATH = "./data/kpfis_op_embed_merged_200.pkl"
DEFAULT_RAW_JSON_PATH = ""

EMBEDDING_MODE = "local_flagembedding"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

USE_FP16 = False
MAX_LENGTH = 1024
BATCH_SIZE = 2

DEFAULT_MILVUS_URI = "./episodic_memory_local.db"

DOC_COLLECTION_NAME = "policy_docs_demo"
MEMORY_COLLECTION_NAME = "episodic_memory_demo"

DOC_TOP_K = 5
MEMORY_TOP_K = 3


# ---------------------------------------------------------------
# Vector dimension safety
# ---------------------------------------------------------------

def force_vector_dim(vec: List[float], target_dim: int):

    if vec is None:
        return [0.0] * target_dim

    cur = len(vec)

    if cur == target_dim:
        return vec

    if cur < target_dim:
        return vec + [0.0] * (target_dim - cur)

    return vec[:target_dim]


# ---------------------------------------------------------------
# Query preprocess
# ---------------------------------------------------------------

def preprocess_query(q: str):

    q = q.replace("있어?", "")
    q = q.replace("있나요?", "")
    q = q.replace("알려줘", "")
    q = q.replace("뭐야", "")
    q = q.strip()

    return q


# ---------------------------------------------------------------
# Chat store
# ---------------------------------------------------------------

CHAT_STORE_DIR = "./chat_store"
os.makedirs(CHAT_STORE_DIR, exist_ok=True)


def get_or_create_chat_id():

    params = st.query_params

    if "chat_id" in params and str(params["chat_id"]).strip():
        return str(params["chat_id"]).strip()

    chat_id = str(uuid.uuid4())[:8]
    st.query_params["chat_id"] = chat_id

    return chat_id


def get_chat_file_path(chat_id: str):

    return os.path.join(CHAT_STORE_DIR, f"chat_{chat_id}.json")


def load_chat_messages(chat_id: str):

    path = get_chat_file_path(chat_id)

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chat_messages(chat_id: str, messages):

    path = get_chat_file_path(chat_id)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------
# Episode
# ---------------------------------------------------------------

@dataclass
class Episode:

    conversation_id: str
    turn_index: int
    user_query: str
    rewritten_query: str
    answer: str
    answer_summary: str
    memory_text: str
    memory_embedding: List[float] = field(default_factory=list)
    score: Optional[float] = None


# ---------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------

class BaseEmbedder(ABC):

    @abstractmethod
    def embed_texts(self, texts):
        pass

    def embed_query(self, text):
        return self.embed_texts([text])[0]

    def embed_memory(self, text):
        return self.embed_texts([text])[0]


class LocalFlagEmbeddingEmbedder(BaseEmbedder):

    def __init__(self):

        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(MODEL_NAME)

    def embed_texts(self, texts):

        out = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return [list(map(float, v)) for v in out["dense_vecs"]]


def build_embedder():

    return LocalFlagEmbeddingEmbedder()


# ---------------------------------------------------------------
# Doc loading
# ---------------------------------------------------------------

def load_docs_from_pkl(path):

    with open(path, "rb") as f:
        raw = pickle.load(f)

    docs = []

    for i, item in enumerate(raw):

        text = item.get("text") or item.get("document")
        embedding = item.get("embedding") or item.get("vector")

        docs.append({
            "id": i,
            "text": text,
            "embedding": embedding,
            "metadata": item
        })

    return docs


def load_docs(bene_path, op_path):

    docs = []

    if bene_path:
        docs.extend(load_docs_from_pkl(bene_path))

    if op_path:

        offset = len(docs)

        op_docs = load_docs_from_pkl(op_path)

        for d in op_docs:
            d["id"] += offset

        docs.extend(op_docs)

    return docs


# ---------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------

def connect_milvus(uri):

    if os.path.exists(uri):
        os.remove(uri)

    return MilvusClient(uri=uri)


# def create_collection(client, name, dim, id_type):

#     if client.has_collection(name):
#         return

#     client.create_collection(
#         collection_name=name,
#         dimension=dim,
#         primary_field_name="id",
#         id_type=id_type,
#         vector_field_name="embedding",
#         metric_type="IP",
#         auto_id=False,
#         enable_dynamic_field=True
#     )

def create_collection(client, name, dim, id_type):

    if client.has_collection(name):
        return

    if id_type == DataType.VARCHAR:

        client.create_collection(
            collection_name=name,
            dimension=dim,
            primary_field_name="id",
            id_type=DataType.VARCHAR,
            max_length=128,          # ⭐ 이게 반드시 필요
            vector_field_name="embedding",
            metric_type="IP",
            auto_id=False,
            enable_dynamic_field=True
        )

    else:

        client.create_collection(
            collection_name=name,
            dimension=dim,
            primary_field_name="id",
            id_type=DataType.INT64,
            vector_field_name="embedding",
            metric_type="IP",
            auto_id=False,
            enable_dynamic_field=True
        )


# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------

def search_docs(client, collection_name, vec):

    results = client.search(
        collection_name=collection_name,
        data=[vec],
        limit=DOC_TOP_K,
        output_fields=["text"]
    )

    docs = []

    for hit in results[0]:

        ent = hit["entity"]

        docs.append({
            "text": ent.get("text"),
            "score": hit.get("distance")
        })

    return docs


# ---------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------

class EpisodicMemoryStore:

    def __init__(self, client, collection, embedder):

        self.client = client
        self.collection = collection
        self.embedder = embedder

    def save(self, episode):

        info = self.client.describe_collection(self.collection)

        dim = info["dimension"]

        emb = force_vector_dim(episode.memory_embedding, dim)

        row = {
            "id": f"{episode.conversation_id}_{episode.turn_index}",
            "conversation_id": episode.conversation_id,
            "memory_text": episode.memory_text,
            "embedding": emb
        }

        self.client.insert(self.collection, [row])

    def recall(self, conversation_id, query):

        info = self.client.describe_collection(self.collection)

        dim = info["dimension"]

        q = self.embedder.embed_memory(query)

        q = force_vector_dim(q, dim)

        res = self.client.search(
            collection_name=self.collection,
            data=[q],
            filter=f'conversation_id == "{conversation_id}"',
            limit=MEMORY_TOP_K,
            output_fields=["memory_text"]
        )

        eps = []

        for h in res[0]:

            ent = h["entity"]

            eps.append(
                Episode(
                    conversation_id=conversation_id,
                    turn_index=0,
                    user_query="",
                    rewritten_query="",
                    answer="",
                    answer_summary="",
                    memory_text=ent.get("memory_text")
                )
            )

        return eps


# ---------------------------------------------------------------
# LLM
# ---------------------------------------------------------------

def call_llm(prompt):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return r.choices[0].message.content


# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------

def rag_pipeline(conversation_id, query, runtime, turn):

    client = runtime["client"]
    embedder = runtime["embedder"]
    memory = runtime["memory"]

    query = preprocess_query(query)

    recalled = memory.recall(conversation_id, query)

    rewritten = query

    search_query = "지원사업 검색: " + rewritten

    q_vec = embedder.embed_query(search_query)

    dim = runtime["vector_dim"]

    q_vec = force_vector_dim(q_vec, dim)

    docs = search_docs(client, DOC_COLLECTION_NAME, q_vec)

    doc_text = "\n".join([d["text"] for d in docs])

    answer = call_llm(f"""
질문:
{query}

문서:
{doc_text}

답변 작성
""")

    mem_text = f"{query} -> {answer[:50]}"

    mem_emb = embedder.embed_memory(mem_text)

    mem_emb = force_vector_dim(mem_emb, dim)

    ep = Episode(
        conversation_id,
        turn,
        query,
        rewritten,
        answer,
        answer[:100],
        mem_text,
        mem_emb
    )

    memory.save(ep)

    return answer


# ---------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------

@st.cache_resource
def build_runtime():

    embedder = build_embedder()

    docs = load_docs(DEFAULT_BENE_PKL_PATH, DEFAULT_OP_PKL_PATH)

    dim = len(docs[0]["embedding"])

    client = connect_milvus(DEFAULT_MILVUS_URI)

    create_collection(client, DOC_COLLECTION_NAME, dim, DataType.INT64)
    create_collection(client, MEMORY_COLLECTION_NAME, dim, DataType.VARCHAR)

    client.insert(DOC_COLLECTION_NAME, docs)

    memory = EpisodicMemoryStore(client, MEMORY_COLLECTION_NAME, embedder)

    return {
        "client": client,
        "embedder": embedder,
        "memory": memory,
        "vector_dim": dim
    }


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------

st.title("🧠 Episodic Memory Demo")

runtime = build_runtime()

chat_id = get_or_create_chat_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "turn" not in st.session_state:
    st.session_state.turn = 0

for m in st.session_state.messages:

    with st.chat_message(m["role"]):
        st.write(m["content"])


prompt = st.chat_input("질문 입력")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        st.session_state.turn += 1

        ans = rag_pipeline(
            chat_id,
            prompt,
            runtime,
            st.session_state.turn
        )

        st.write(ans)

        st.session_state.messages.append({
            "role":"assistant",
            "content":ans
        })
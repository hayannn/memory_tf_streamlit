import os
import json
import pickle
import uuid
from dataclasses import dataclass, field
from typing import List, Optional
from abc import ABC, abstractmethod

import streamlit as st
from pymilvus import MilvusClient, DataType
from openai import OpenAI


# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
DEFAULT_BENE_PKL_PATH = "./data/kpfis_bene_embed_merged_200.pkl"
DEFAULT_OP_PKL_PATH = "./data/kpfis_op_embed_merged_200.pkl"

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
def force_vector_dim(vec: List[float], target_dim: int) -> List[float]:
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
def preprocess_query(q: str) -> str:
    q = q.replace("있어?", "")
    q = q.replace("있나요?", "")
    q = q.replace("알려줘", "")
    q = q.replace("뭐야", "")
    return q.strip()


# ---------------------------------------------------------------
# Chat store
# ---------------------------------------------------------------
CHAT_STORE_DIR = "./chat_store"
os.makedirs(CHAT_STORE_DIR, exist_ok=True)


def get_or_create_chat_id() -> str:
    params = st.query_params

    if "chat_id" in params and str(params["chat_id"]).strip():
        return str(params["chat_id"]).strip()

    chat_id = str(uuid.uuid4())[:8]
    st.query_params["chat_id"] = chat_id
    return chat_id


def get_chat_file_path(chat_id: str) -> str:
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
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_memory(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class LocalFlagEmbeddingEmbedder(BaseEmbedder):
    def __init__(self):
        from FlagEmbedding import BGEM3FlagModel

        self.model_name = MODEL_NAME
        self.model = BGEM3FlagModel(
            MODEL_NAME,
            use_fp16=USE_FP16
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        out = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return [list(map(float, v)) for v in out["dense_vecs"]]


def build_embedder() -> BaseEmbedder:
    if EMBEDDING_MODE != "local_flagembedding":
        raise ValueError(f"지원하지 않는 EMBEDDING_MODE 입니다: {EMBEDDING_MODE}")
    return LocalFlagEmbeddingEmbedder()


# ---------------------------------------------------------------
# Doc loading
# ---------------------------------------------------------------
def load_docs_from_pkl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    with open(path, "rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"PKL 내부 데이터는 list여야 합니다: {path}")

    docs = []

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue

        text = item.get("text") or item.get("document") or ""
        embedding = item.get("embedding") or item.get("vector")

        if embedding is None:
            continue

        docs.append({
            "id": i,
            "text": str(text),
            "embedding": embedding,
            "metadata": item
        })

    return docs


def load_docs(bene_path: str, op_path: str):
    docs = []

    if bene_path:
        docs.extend(load_docs_from_pkl(bene_path))

    if op_path:
        offset = len(docs)
        op_docs = load_docs_from_pkl(op_path)

        for d in op_docs:
            d["id"] += offset

        docs.extend(op_docs)

    if not docs:
        raise ValueError("로드된 문서가 없습니다.")

    return docs


# ---------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------
def connect_milvus(uri: str) -> MilvusClient:
    if os.path.exists(uri):
        os.remove(uri)
    return MilvusClient(uri=uri)


def create_collection(client: MilvusClient, name: str, dim: int, id_type):
    if client.has_collection(name):
        return

    if id_type == DataType.VARCHAR:
        client.create_collection(
            collection_name=name,
            dimension=dim,
            primary_field_name="id",
            id_type=DataType.VARCHAR,
            max_length=128,
            vector_field_name="embedding",
            metric_type="IP",
            auto_id=False,
            enable_dynamic_field=True,
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
            enable_dynamic_field=True,
        )


# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------
def search_docs(client: MilvusClient, collection_name: str, vec: List[float]):
    results = client.search(
        collection_name=collection_name,
        data=[vec],
        limit=DOC_TOP_K,
        output_fields=["text"],
    )

    docs = []

    for hit in results[0]:
        ent = hit.get("entity", {})
        docs.append({
            "text": ent.get("text", ""),
            "score": hit.get("distance", hit.get("score")),
        })

    return docs


# ---------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------
class EpisodicMemoryStore:
    def __init__(self, client: MilvusClient, collection: str, embedder: BaseEmbedder, vector_dim: int):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.vector_dim = vector_dim

    def save(self, episode: Episode):
        emb = force_vector_dim(episode.memory_embedding, self.vector_dim)

        row = {
            "id": f"{episode.conversation_id}_{episode.turn_index}",
            "conversation_id": episode.conversation_id,
            "memory_text": episode.memory_text,
            "embedding": emb,
        }

        self.client.insert(
            collection_name=self.collection,
            data=[row],
        )

    def recall(self, conversation_id: str, query: str):
        q = self.embedder.embed_memory(query)
        q = force_vector_dim(q, self.vector_dim)

        res = self.client.search(
            collection_name=self.collection,
            data=[q],
            filter=f'conversation_id == "{conversation_id}"',
            limit=MEMORY_TOP_K,
            output_fields=["memory_text"],
        )

        eps = []

        for h in res[0]:
            ent = h.get("entity", {})
            eps.append(
                Episode(
                    conversation_id=conversation_id,
                    turn_index=0,
                    user_query="",
                    rewritten_query="",
                    answer="",
                    answer_summary="",
                    memory_text=ent.get("memory_text", ""),
                )
            )

        return eps


# ---------------------------------------------------------------
# LLM
# ---------------------------------------------------------------
def call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    client = OpenAI(api_key=api_key)

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return r.choices[0].message.content


# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------
def rag_pipeline(conversation_id: str, query: str, runtime, turn: int):
    client = runtime["client"]
    embedder = runtime["embedder"]
    memory = runtime["memory"]

    query = preprocess_query(query)

    recalled = memory.recall(conversation_id, query)
    recalled_text = "\n".join([ep.memory_text for ep in recalled]) if recalled else "없음"

    rewritten = query
    search_query = "지원사업 검색: " + rewritten

    q_vec = embedder.embed_query(search_query)
    q_vec = force_vector_dim(q_vec, runtime["vector_dim"])

    docs = search_docs(client, DOC_COLLECTION_NAME, q_vec)
    doc_text = "\n".join([d["text"] for d in docs if d.get("text")])

    answer = call_llm(f"""
사용자 질문:
{query}

이전 메모:
{recalled_text}

검색 문서:
{doc_text}

규칙:
- 문서 기반으로만 답변하세요.
- 문서에 없는 내용은 추측하지 마세요.
- 한국어로 간결하게 답하세요.

답변:
""")

    mem_text = f"{query} -> {answer[:80]}"
    mem_emb = embedder.embed_memory(mem_text)
    mem_emb = force_vector_dim(mem_emb, runtime["vector_dim"])

    ep = Episode(
        conversation_id=conversation_id,
        turn_index=turn,
        user_query=query,
        rewritten_query=rewritten,
        answer=answer,
        answer_summary=answer[:100],
        memory_text=mem_text,
        memory_embedding=mem_emb,
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

    first_emb = docs[0].get("embedding")
    if first_emb is None:
        raise ValueError("첫 문서에 embedding 이 없습니다.")

    dim = len(first_emb)

    client = connect_milvus(DEFAULT_MILVUS_URI)

    create_collection(client, DOC_COLLECTION_NAME, dim, DataType.INT64)
    create_collection(client, MEMORY_COLLECTION_NAME, dim, DataType.VARCHAR)

    client.insert(
        collection_name=DOC_COLLECTION_NAME,
        data=docs,
    )

    memory = EpisodicMemoryStore(
        client=client,
        collection=MEMORY_COLLECTION_NAME,
        embedder=embedder,
        vector_dim=dim,
    )

    return {
        "client": client,
        "embedder": embedder,
        "memory": memory,
        "vector_dim": dim,
    }


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.title("🧠 Episodic Memory Demo")

runtime = build_runtime()
chat_id = get_or_create_chat_id()

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_messages(chat_id)

if "turn" not in st.session_state:
    st.session_state.turn = sum(1 for m in st.session_state.messages if m["role"] == "user")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("질문 입력")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_messages(chat_id, st.session_state.messages)

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            st.session_state.turn += 1

            ans = rag_pipeline(
                chat_id,
                prompt,
                runtime,
                st.session_state.turn,
            )

            st.write(ans)

            st.session_state.messages.append({
                "role": "assistant",
                "content": ans,
            })
            save_chat_messages(chat_id, st.session_state.messages)

        except Exception as e:
            import traceback
            st.error(f"실행 오류: {e}")
            st.code(traceback.format_exc())
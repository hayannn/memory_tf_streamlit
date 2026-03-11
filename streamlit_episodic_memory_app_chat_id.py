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

MODEL_NAME = "BAAI/bge-small-en-v1.5"

DOC_TOP_K = 10
MEMORY_TOP_K = 3

DEFAULT_MILVUS_URI = "./episodic_memory_local.db"

DOC_COLLECTION_NAME = "policy_docs_demo"
MEMORY_COLLECTION_NAME = "episodic_memory_demo"

BATCH_SIZE = 4
MAX_LENGTH = 512
USE_FP16 = False

CHAT_STORE_DIR = "./chat_store"
os.makedirs(CHAT_STORE_DIR, exist_ok=True)

TRACK_KEYWORDS = [
    "청년","중장년","서울","경기","취업","교육",
    "창업","주거","접수","마감","지원사업"
]


# ---------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------

def preprocess_query(q: str):

    if not q:
        return ""

    q = q.replace("있어?", "")
    q = q.replace("있나요?", "")
    q = q.replace("알려줘", "")
    q = q.replace("뭐야", "")
    q = q.replace("좀", "")
    q = q.strip()

    return q


# ---------------------------------------------------------------
# Chat Store
# ---------------------------------------------------------------

def get_or_create_chat_id():

    params = st.query_params

    if "chat_id" in params and str(params["chat_id"]).strip():
        return str(params["chat_id"])

    chat_id = str(uuid.uuid4())[:8]

    st.query_params["chat_id"] = chat_id

    return chat_id


def get_chat_file_path(chat_id):

    return os.path.join(CHAT_STORE_DIR, f"chat_{chat_id}.json")


def load_chat_messages(chat_id):

    path = get_chat_file_path(chat_id)

    if not os.path.exists(path):
        return []

    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)


def save_chat_messages(chat_id, messages):

    path = get_chat_file_path(chat_id)

    with open(path,"w",encoding="utf-8") as f:
        json.dump(messages,f,ensure_ascii=False,indent=2)


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
        raise NotImplementedError

    def embed_query(self, text):
        return self.embed_texts([text])[0]

    def embed_memory(self, text):
        return self.embed_texts([text])[0]


class LocalFlagEmbeddingEmbedder(BaseEmbedder):

    def __init__(self):

        from FlagEmbedding import FlagModel

        self.model = FlagModel(
            MODEL_NAME,
            use_fp16=USE_FP16
        )

    def embed_texts(self, texts):

        dense = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH
        )

        return [list(map(float,v)) for v in dense]


def build_embedder():

    return LocalFlagEmbeddingEmbedder()


# ---------------------------------------------------------------
# Document Load
# ---------------------------------------------------------------

def load_docs_from_pkl(path):

    with open(path,"rb") as f:
        raw = pickle.load(f)

    docs=[]

    for i,item in enumerate(raw):

        docs.append({
            "id": int(item.get("id",i)),
            "text": str(item.get("text","")),
            "embedding": item.get("embedding"),
            "metadata": item
        })

    return docs


def load_docs(bene_path, op_path):

    docs=[]

    if bene_path:

        docs.extend(load_docs_from_pkl(bene_path))

    if op_path:

        op=load_docs_from_pkl(op_path)

        for d in op:
            d["id"] += 1_000_000

        docs.extend(op)

    return docs


# ---------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------

def connect_milvus(uri):

    return MilvusClient(uri=uri)


def create_docs_collection_if_needed(client, dim):

    if client.has_collection(DOC_COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=DOC_COLLECTION_NAME,
        dimension=dim,
        primary_field_name="id",
        id_type=DataType.INT64,
        vector_field_name="embedding",
        metric_type="IP",
        auto_id=False,
        enable_dynamic_field=True
    )


# ---------------------------------------------------------------
# Vector Search
# ---------------------------------------------------------------

def search_docs(client, query_vec, top_k):

    results = client.search(
        collection_name=DOC_COLLECTION_NAME,
        data=[query_vec],
        limit=top_k,
        output_fields=["id","text","metadata"],
        search_params={"metric_type":"IP"}
    )

    docs=[]

    for hit in results[0]:

        ent = hit["entity"]

        docs.append({
            "id": ent["id"],
            "text": ent["text"],
            "metadata": ent["metadata"],
            "score": hit["distance"]
        })

    return docs


# ---------------------------------------------------------------
# LLM
# ---------------------------------------------------------------

def call_llm(prompt, system):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":prompt}
        ],
        temperature=0.2
    )

    return res.choices[0].message.content.strip()


# ---------------------------------------------------------------
# Query Rewrite
# ---------------------------------------------------------------

def rewrite_query(user_query):

    system="검색 질의 재작성기"

    prompt=f"""
다음 질문을 검색용 질의로 간단히 재작성하라.

질문:
{user_query}

질의:
"""

    return call_llm(prompt,system)


# ---------------------------------------------------------------
# Answer Prompt
# ---------------------------------------------------------------

def build_answer_prompt(user_query, docs):

    doc_texts=[d["text"] for d in docs[:5]]

    docs_block="\n".join([f"- {t}" for t in doc_texts]) if doc_texts else "- 없음"

    return f"""사용자 질문에 대해 검색 문서만 근거로 답변하라.

규칙:
- 반드시 아래 검색 문서 내용만 사용하라
- 문서에 없는 내용은 절대 추측하지 마라
- 관련 사업이 여러 개면 bullet로 정리하라
- 사용자가 요청한 필터 조건을 가능한 범위에서 반영하라
- 문서를 찾지 못하면 "조건에 맞는 문서를 찾지 못했습니다."라고 답하라

사용자 질문:
{user_query}

검색 문서:
{docs_block}

답변:"""


def generate_answer(query, docs):

    if not docs:
        return "조건에 맞는 문서를 찾지 못했습니다."

    prompt = build_answer_prompt(query, docs)

    system = "정책 정보 요약 도우미"

    return call_llm(prompt, system)


# ---------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------

def rag_pipeline(chat_id, user_query, client, embedder, turn):

    user_query = preprocess_query(user_query)

    rewritten = rewrite_query(user_query)

    if not rewritten:
        rewritten = user_query

    final_query = rewritten.strip()

    if not final_query:
        final_query = user_query

    search_query = "지원사업 정책 검색: " + final_query

    q_vec = embedder.embed_query(search_query)

    docs = search_docs(client, q_vec, DOC_TOP_K)

    answer = generate_answer(user_query, docs)

    return {
        "answer":answer,
        "docs":docs,
        "query":final_query
    }


# ---------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------

@st.cache_resource
def build_runtime():

    embedder = build_embedder()

    docs = load_docs(DEFAULT_BENE_PKL_PATH, DEFAULT_OP_PKL_PATH)

    dim = len(docs[0]["embedding"])

    client = connect_milvus(DEFAULT_MILVUS_URI)

    create_docs_collection_if_needed(client, dim)

    return {
        "client":client,
        "embedder":embedder
    }


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------

st.set_page_config(page_title="Episodic Memory Demo",layout="wide")

st.title("🧠 Episodic Memory + Milvus + BGE")

runtime = build_runtime()

chat_id = get_or_create_chat_id()

messages = load_chat_messages(chat_id)

for m in messages:

    with st.chat_message(m["role"]):
        st.markdown(m["content"])


prompt = st.chat_input("질문을 입력하세요")

if prompt:

    messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    res = rag_pipeline(
        chat_id,
        prompt,
        runtime["client"],
        runtime["embedder"],
        len(messages)
    )

    with st.chat_message("assistant"):
        st.markdown(res["answer"])

    messages.append({
        "role":"assistant",
        "content":res["answer"]
    })

    save_chat_messages(chat_id,messages)
import os
import json
import pickle
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import streamlit as st
from pymilvus import MilvusClient, DataType, connections
from openai import OpenAI

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
DEFAULT_BENE_PKL_PATH = "/home/dlgkd/dev2/TF/data/kpfis_bene_embed_merged_200.pkl"
DEFAULT_OP_PKL_PATH = "/home/dlgkd/dev2/TF/data/kpfis_op_embed_merged_200.pkl"
DEFAULT_RAW_JSON_PATH = ""

EMBEDDING_MODE = "local_flagembedding"
MODEL_NAME = "BAAI/bge-m3"
USE_FP16 = False
MAX_LENGTH = 1024
BATCH_SIZE = 2

DEFAULT_MILVUS_URI = "./episodic_memory_local.db"
DOC_COLLECTION_NAME = "policy_docs_demo"
MEMORY_COLLECTION_NAME = "episodic_memory_demo"

DOC_TOP_K = 5
MEMORY_TOP_K = 3

TRACK_KEYWORDS = [
    "청년", "중장년", "서울", "경기", "취업", "교육",
    "창업", "주거", "접수 중", "마감", "지원사업"
]

CHAT_STORE_DIR = "./chat_store"
os.makedirs(CHAT_STORE_DIR, exist_ok=True)


def preprocess_query(q: str) -> str:
    q = q.replace("있어?", "")
    q = q.replace("있나요?", "")
    q = q.replace("알려줘", "")
    q = q.replace("뭐야", "")
    q = q.strip()
    return q


# ---------------------------------------------------------------
# Chat ID / Chat Store
# ---------------------------------------------------------------
def get_or_create_chat_id() -> str:
    params = st.query_params

    if "chat_id" in params and str(params["chat_id"]).strip():
        return str(params["chat_id"]).strip()

    chat_id = str(uuid.uuid4())[:8]
    st.query_params["chat_id"] = chat_id
    return chat_id


def get_chat_file_path(chat_id: str) -> str:
    return os.path.join(CHAT_STORE_DIR, f"chat_{chat_id}.json")


def load_chat_messages(chat_id: str) -> List[Dict[str, Any]]:
    path = get_chat_file_path(chat_id)
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chat_messages(chat_id: str, messages: List[Dict[str, Any]]):
    path = get_chat_file_path(chat_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------
# 데이터 구조
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
# 임베더
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
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        use_fp16: bool = USE_FP16,
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
    ):
        from FlagEmbedding import BGEM3FlagModel

        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        return [list(map(float, vec)) for vec in dense]


def build_embedder(mode: str = EMBEDDING_MODE) -> BaseEmbedder:
    if mode == "local_flagembedding":
        return LocalFlagEmbeddingEmbedder()
    raise ValueError(f"지원하지 않는 EMBEDDING_MODE 입니다: {mode}")


# ---------------------------------------------------------------
# 문서 로드
# ---------------------------------------------------------------
def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def normalize_docs(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []

    for idx, item in enumerate(raw_docs):
        if not isinstance(item, dict):
            raise ValueError(f"{idx}번 문서는 dict여야 합니다. type={type(item)}")

        metadata = item.get("metadata", {})
        doc_id = None
        text = None
        embedding = None

        if "vector" in item and "document" in item:
            doc_id = metadata.get("id", item.get("record_id", idx))
            text = item.get("document")
            embedding = item.get("vector")
        elif "embedding" in item and "text" in item:
            doc_id = item.get("id", idx)
            text = item.get("text")
            embedding = item.get("embedding")
            metadata = item
        elif "text" in item or "document" in item:
            doc_id = item.get("id", idx)
            text = item.get("text", item.get("document"))
            embedding = item.get("embedding")
            metadata = item
        else:
            raise ValueError(f"{idx}번 문서 포맷을 인식할 수 없습니다: keys={list(item.keys())}")

        docs.append({
            "id": _safe_int(doc_id, idx),
            "text": str(text) if text is not None else "",
            "embedding": embedding,
            "metadata": metadata if isinstance(metadata, dict) else {},
        })

    if not docs:
        raise ValueError("문서가 비어 있습니다.")

    return docs


def load_docs_from_pkl(pkl_path: str) -> List[Dict[str, Any]]:
    with open(pkl_path, "rb") as f:
        raw_docs = pickle.load(f)

    if not isinstance(raw_docs, list):
        raise ValueError("pkl 내부 데이터는 list 형태여야 합니다.")

    return normalize_docs(raw_docs)


def load_docs_from_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)
    elif p.suffix.lower() == ".jsonl":
        raw_docs = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_docs.append(json.loads(line))
    else:
        raise ValueError("RAW_JSON_PATH는 .json 또는 .jsonl 이어야 합니다.")

    if not isinstance(raw_docs, list):
        raise ValueError("json/jsonl 로드 결과는 list 형태여야 합니다.")

    return normalize_docs(raw_docs)


def load_docs(bene_pkl_path: str, op_pkl_path: str, raw_json_path: str = "") -> List[Dict[str, Any]]:
    if raw_json_path:
        return load_docs_from_json(raw_json_path)

    docs = []

    if bene_pkl_path:
        bene_docs = load_docs_from_pkl(bene_pkl_path)
        docs.extend(bene_docs)

    if op_pkl_path:
        op_docs = load_docs_from_pkl(op_pkl_path)
        for d in op_docs:
            d["id"] += 1_000_000
        docs.extend(op_docs)

    if not docs:
        raise ValueError("로드된 문서가 없습니다.")

    return docs


def enrich_docs_with_embeddings_if_needed(
    docs: List[Dict[str, Any]],
    embedder: BaseEmbedder,
    batch_size: int = BATCH_SIZE,
) -> List[Dict[str, Any]]:
    needs = [i for i, d in enumerate(docs) if not d.get("embedding")]
    if not needs:
        return docs

    for start in range(0, len(needs), batch_size):
        idxs = needs[start:start + batch_size]
        texts = [docs[i]["text"] for i in idxs]
        vecs = embedder.embed_texts(texts)
        for i, v in zip(idxs, vecs):
            docs[i]["embedding"] = v
    return docs


def infer_vector_dim_from_docs_or_embedder(docs: List[Dict[str, Any]], embedder: BaseEmbedder) -> int:
    for d in docs:
        emb = d.get("embedding")
        if emb:
            return len(emb)
    return len(embedder.embed_query("테스트 질의"))


# ---------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------
def connect_milvus_lite(uri: str) -> MilvusClient:
    connections.disconnect("default")
    connections.connect(alias="default", uri=uri)
    return MilvusClient(uri=uri)


def drop_collection_if_exists(client: MilvusClient, collection_name: str):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)


def create_docs_collection_if_needed(client: MilvusClient, collection_name: str, dim: int):
    if client.has_collection(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        primary_field_name="id",
        id_type=DataType.INT64,
        vector_field_name="embedding",
        metric_type="COSINE",
        auto_id=False,
        enable_dynamic_field=True,
    )


def create_memory_collection_if_needed(client: MilvusClient, collection_name: str, dim: int):
    if client.has_collection(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        primary_field_name="id",
        id_type=DataType.VARCHAR,
        max_length=128,
        vector_field_name="embedding",
        metric_type="COSINE",
        auto_id=False,
        enable_dynamic_field=True,
    )


def insert_docs_to_milvus(client: MilvusClient, collection_name: str, docs: List[Dict[str, Any]]):
    rows = []
    for doc in docs:
        rows.append({
            "id": int(doc["id"]),
            "text": str(doc["text"]),
            "embedding": doc["embedding"],
            "metadata": doc.get("metadata", {}),
        })
    client.insert(collection_name=collection_name, data=rows)


def clear_collection_data(client: MilvusClient, collection_name: str):
    if not client.has_collection(collection_name):
        return
    filt = "id != ''" if collection_name == MEMORY_COLLECTION_NAME else "id >= 0"
    client.delete(collection_name=collection_name, filter=filt)


def clear_memory_by_conversation(client: MilvusClient, collection_name: str, conversation_id: str):
    if not client.has_collection(collection_name):
        return
    client.delete(
        collection_name=collection_name,
        filter=f'conversation_id == "{conversation_id}"'
    )


def search_docs(
    client: MilvusClient,
    collection_name: str,
    query_embedding: List[float],
    top_k: int = DOC_TOP_K,
) -> List[Dict[str, Any]]:
    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=top_k,
        output_fields=["id", "text", "metadata"],
        search_params={"metric_type": "COSINE", "params": {}},
    )

    docs = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        docs.append({
            "id": entity.get("id"),
            "text": entity.get("text"),
            "metadata": entity.get("metadata"),
            "score": hit.get("distance", hit.get("score")),
        })
    return docs


# ---------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------
class EpisodicMemoryStore:
    def __init__(self, client: MilvusClient, collection_name: str, embedder: BaseEmbedder):
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    def _build_memory_id(self, conversation_id: str, turn_index: int) -> str:
        return f"{conversation_id}_turn_{turn_index}"

    def save(self, episode: Episode):
        row = {
            "id": self._build_memory_id(episode.conversation_id, episode.turn_index),
            "conversation_id": episode.conversation_id,
            "turn_index": episode.turn_index,
            "user_query": episode.user_query,
            "rewritten_query": episode.rewritten_query,
            "answer": episode.answer,
            "answer_summary": episode.answer_summary,
            "memory_text": episode.memory_text,
            "embedding": episode.memory_embedding,
        }
        self.client.insert(collection_name=self.collection_name, data=[row])

    def recall(
        self,
        conversation_id: str,
        user_query: str,
        top_k: int = MEMORY_TOP_K
    ) -> List[Episode]:
        q_vec = self.embedder.embed_memory(user_query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[q_vec],
            limit=top_k,
            filter=f'conversation_id == "{conversation_id}"',
            output_fields=[
                "conversation_id",
                "turn_index",
                "user_query",
                "rewritten_query",
                "answer",
                "answer_summary",
                "memory_text",
            ],
            search_params={"metric_type": "COSINE", "params": {}},
        )

        recalled = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            recalled.append(
                Episode(
                    conversation_id=entity.get("conversation_id"),
                    turn_index=entity.get("turn_index"),
                    user_query=entity.get("user_query"),
                    rewritten_query=entity.get("rewritten_query"),
                    answer=entity.get("answer"),
                    answer_summary=entity.get("answer_summary"),
                    memory_text=entity.get("memory_text"),
                    memory_embedding=[],
                    score=hit.get("distance", hit.get("score")),
                )
            )
        return recalled


# ---------------------------------------------------------------
# LLM 유틸
# ---------------------------------------------------------------
def _clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _dedup_texts(texts: List[str]) -> List[str]:
    seen = set()
    result = []
    for text in texts:
        cleaned = _clean_text(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _extract_doc_texts(retrieved_docs: List[Dict[str, Any]], max_docs: int = 5) -> List[str]:
    texts = []
    for doc in retrieved_docs[:max_docs]:
        txt = _clean_text(doc.get("text", ""))
        if txt:
            texts.append(txt)
    return _dedup_texts(texts)


def _match_keywords(user_query: str, texts: List[str]) -> List[str]:
    merged = user_query + " " + " ".join(texts)
    return [kw for kw in TRACK_KEYWORDS if kw in merged]


def _call_llm(prompt: str, system_prompt: str, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _build_rewrite_prompt(user_query: str, recalled_eps: List[Episode]) -> str:
    memory_context = "\n".join([f"- {ep.memory_text}" for ep in recalled_eps]) if recalled_eps else "- 없음"
    return f"""사용자의 현재 질문을 검색에 유리한 질의로 재작성하라.

목표:
- 이전 대화 맥락을 반영한다.
- 사용자의 조건(대상, 지역, 분야, 상태)을 최대한 보존한다.
- 길게 설명하지 말고, 검색용 질의 한 문장만 출력한다.
- 없는 조건을 추측해서 추가하지 마라.
- 애매한 대명사(예: 다시, 그거, 거기, 그쪽)는 이전 맥락을 반영해 구체화하라.

이전 대화 맥락:
{memory_context}

현재 질문:
{user_query}

재작성 질의:"""


def rewrite_query(user_query: str, recalled_eps: List[Episode]) -> str:
    system_prompt = "너는 검색 질의 재작성기다. 설명 없이 재작성 질의 한 문장만 출력한다."
    prompt = _build_rewrite_prompt(user_query, recalled_eps)
    return _call_llm(prompt, system_prompt)


def _build_answer_prompt(user_query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    doc_texts = _extract_doc_texts(retrieved_docs, max_docs=5)
    docs_block = "\n".join([f"- {t}" for t in doc_texts]) if doc_texts else "- 없음"
    return f"""사용자 질문에 대해 검색 문서만 근거로 답변하라.

규칙:
- 문서에 없는 내용은 추측하지 마라.
- 문서 기반으로 핵심만 자연스럽게 정리하라.
- 관련 사업이 여러 개면 bullet로 나눠도 된다.
- 사용자가 필터를 요청한 경우(예: 서울, 취업, 마감 제외) 가능한 범위에서 반영하라.
- 문서를 찾지 못하면 찾지 못했다고 말하라.

사용자 질문:
{user_query}

검색 문서:
{docs_block}

답변:"""


def generate_answer(user_query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "조건에 맞는 문서를 찾지 못했습니다."
    system_prompt = "너는 정책/사업 정보 요약 도우미다. 문서 근거 중심으로 한국어로 답한다."
    prompt = _build_answer_prompt(user_query, retrieved_docs)
    return _call_llm(prompt, system_prompt)


def summarize_memory(user_query: str, answer: str) -> str:
    found = _match_keywords(user_query, [answer])
    if found:
        return "사용자 관심 조건: " + ", ".join(found)
    return f"사용자는 '{user_query}' 관련 정보를 찾고 있다."


# ---------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------
def rag_pipeline(
    conversation_id: str,
    user_query: str,
    client: MilvusClient,
    embedder: BaseEmbedder,
    memory_store: EpisodicMemoryStore,
    turn_index: int,
    doc_collection_name: str = DOC_COLLECTION_NAME,
    top_k: int = DOC_TOP_K,
):

    # 1️⃣ query normalize
    user_query = preprocess_query(user_query)

    # 2️⃣ recall episodic memory
    recalled_eps = memory_store.recall(
        conversation_id=conversation_id,
        user_query=user_query,
        top_k=MEMORY_TOP_K,
    )

    # 3️⃣ rewrite query
    rewritten_query = rewrite_query(user_query, recalled_eps)

    if not rewritten_query or not str(rewritten_query).strip():
        rewritten_query = user_query

    final_query_for_search = rewritten_query.strip()

    # 4️⃣ rewrite fallback
    if not final_query_for_search:
        final_query_for_search = user_query

    # 5️⃣ BGE instruction (검색 recall 증가)
    search_query = "지원사업 정책 검색: " + final_query_for_search

    # 6️⃣ embedding
    q_vec = embedder.embed_query(search_query)

    # 7️⃣ vector search
    retrieved_docs = search_docs(
        client,
        doc_collection_name,
        q_vec,
        top_k=top_k
    )

    # 8️⃣ answer generation
    answer = generate_answer(user_query, retrieved_docs)

    # 9️⃣ memory summary
    memory_text = summarize_memory(user_query, answer)

    memory_embedding = embedder.embed_memory(memory_text)

    answer_summary = " ".join(answer.split())[:120]

    episode = Episode(
        conversation_id=conversation_id,
        turn_index=turn_index,
        user_query=user_query,
        rewritten_query=final_query_for_search,
        answer=answer,
        answer_summary=answer_summary,
        memory_text=memory_text,
        memory_embedding=memory_embedding,
    )

    memory_store.save(episode)

    recalled_memory_list = []
    for ep in recalled_eps:
        recalled_memory_list.append({
            "conversation_id": ep.conversation_id,
            "turn_index": ep.turn_index,
            "memory_text": ep.memory_text,
            "score": getattr(ep, "score", None),
        })

    return {
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "user_query": user_query,
        "recalled_memory": recalled_memory_list,
        "rewritten_query": rewritten_query,
        "final_query_for_search": final_query_for_search,
        "retrieved_docs": retrieved_docs,
        "answer": answer,
        "saved_memory": memory_text,
    }


# ---------------------------------------------------------------
# 캐시 초기화
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_runtime(
    bene_pkl_path: str,
    op_pkl_path: str,
    raw_json_path: str,
    milvus_uri: str,
    reindex_docs: bool,
    reset_memory: bool,
):
    embedder = build_embedder(EMBEDDING_MODE)
    docs = load_docs(bene_pkl_path, op_pkl_path, raw_json_path)
    docs = enrich_docs_with_embeddings_if_needed(docs=docs, embedder=embedder, batch_size=BATCH_SIZE)
    vector_dim = infer_vector_dim_from_docs_or_embedder(docs, embedder)

    client = connect_milvus_lite(milvus_uri)

    if reindex_docs:
        drop_collection_if_exists(client, DOC_COLLECTION_NAME)
    create_docs_collection_if_needed(client, DOC_COLLECTION_NAME, vector_dim)

    if reindex_docs:
        insert_docs_to_milvus(client, DOC_COLLECTION_NAME, docs)
    else:
        existing_count = int(client.get_collection_stats(DOC_COLLECTION_NAME).get("row_count", 0))
        if existing_count == 0:
            insert_docs_to_milvus(client, DOC_COLLECTION_NAME, docs)

    if reset_memory:
        drop_collection_if_exists(client, MEMORY_COLLECTION_NAME)
    create_memory_collection_if_needed(client, MEMORY_COLLECTION_NAME, vector_dim)

    memory_store = EpisodicMemoryStore(
        client=client,
        collection_name=MEMORY_COLLECTION_NAME,
        embedder=embedder,
    )

    return {
        "client": client,
        "embedder": embedder,
        "memory_store": memory_store,
        "doc_count": len(docs),
        "vector_dim": vector_dim,
        "milvus_uri": milvus_uri,
    }


def reset_chat_state(chat_id: str):
    st.session_state.messages = []
    st.session_state.turn_index = 0
    save_chat_messages(chat_id, [])


def reset_memory_collection(runtime: Dict[str, Any], chat_id: str):
    clear_memory_by_conversation(runtime["client"], MEMORY_COLLECTION_NAME, chat_id)
    reset_chat_state(chat_id)


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Episodic Memory Demo", page_icon="🧠", layout="wide")
st.title("🧠 Episodic Memory + Milvus + FlagEmbedding")

chat_id = get_or_create_chat_id()

if "chat_id" not in st.session_state or st.session_state.chat_id != chat_id:
    st.session_state.chat_id = chat_id
    st.session_state.messages = load_chat_messages(chat_id)
    st.session_state.turn_index = sum(1 for m in st.session_state.messages if m["role"] == "user")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_messages(chat_id)

if "turn_index" not in st.session_state:
    st.session_state.turn_index = sum(1 for m in st.session_state.messages if m["role"] == "user")

with st.sidebar:
    st.subheader("설정")
    bene_pkl_path = st.text_input("BENE PKL 경로", value=DEFAULT_BENE_PKL_PATH)
    op_pkl_path = st.text_input("OP PKL 경로", value=DEFAULT_OP_PKL_PATH)
    raw_json_path = st.text_input("RAW JSON/JSONL 경로 (선택)", value=DEFAULT_RAW_JSON_PATH)
    milvus_uri = st.text_input("Milvus URI", value=DEFAULT_MILVUS_URI)
    reindex_docs = st.checkbox("문서 컬렉션 재색인", value=False)
    reset_memory = st.checkbox("앱 시작 시 메모리 컬렉션 초기화", value=False)

    st.markdown("---")
    st.subheader("현재 채팅 링크")
    st.code(f"?chat_id={chat_id}", language=None)
    st.caption("같은 chat_id로 접속하면 같은 채팅/메모리를 사용합니다.")

    if st.button("새 채팅 시작"):
        new_chat_id = str(uuid.uuid4())[:8]
        st.query_params["chat_id"] = new_chat_id
        st.session_state.chat_id = new_chat_id
        st.session_state.messages = []
        st.session_state.turn_index = 0
        save_chat_messages(new_chat_id, [])
        st.rerun()

    st.markdown("---")
    st.write("샘플 질문")
    sample_queries = [
        "청년 대상 사업 있어?",
        "서울 기준으로 다시 알려줘",
        "취업 관련만 보고 싶어",
        "마감된 건 빼줘",
        "보조사업 중에서 청년 대상만 보여줘",
        "공모 사업 정보도 추가해줘",
    ]

    for idx, q in enumerate(sample_queries):
        if st.button(q, key=f"sample_{idx}"):
            st.session_state["pending_query"] = q

runtime = None
error_box = st.empty()

try:
    runtime = build_runtime(
        bene_pkl_path=bene_pkl_path,
        op_pkl_path=op_pkl_path,
        raw_json_path=raw_json_path.strip(),
        milvus_uri=milvus_uri,
        reindex_docs=reindex_docs,
        reset_memory=reset_memory,
    )
except Exception as e:
    error_box.error(f"초기화 실패: {e}")

if runtime:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("문서 수", runtime["doc_count"])
    with col2:
        st.metric("벡터 차원", runtime["vector_dim"])
    with col3:
        st.write(f"Milvus: `{runtime['milvus_uri']}`")

    st.info(f"현재 chat_id: {chat_id}")

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("채팅 기록 초기화"):
            reset_chat_state(chat_id)
            st.rerun()
    with btn2:
        if st.button("메모리 컬렉션 초기화"):
            reset_memory_collection(runtime, chat_id)
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("debug"):
            with st.expander("디버그 정보"):
                st.json(msg["debug"])

prompt = st.chat_input("질문을 입력하세요")
if not prompt and st.session_state.get("pending_query"):
    prompt = st.session_state.pop("pending_query")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_messages(chat_id, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    if not runtime:
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                st.session_state.turn_index += 1
                res = rag_pipeline(
                    conversation_id=chat_id,
                    user_query=prompt,
                    client=runtime["client"],
                    embedder=runtime["embedder"],
                    memory_store=runtime["memory_store"],
                    turn_index=st.session_state.turn_index,
                    top_k=DOC_TOP_K,
                )

                st.markdown(res["answer"])
                debug = {
                    "conversation_id": res["conversation_id"],
                    "turn_index": res["turn_index"],
                    "recalled_memory": res["recalled_memory"],
                    "rewritten_query": res["rewritten_query"],
                    "final_query_for_search": res["final_query_for_search"],
                    "retrieved_docs": res["retrieved_docs"],
                    "saved_memory": res["saved_memory"],
                }
                with st.expander("디버그 정보"):
                    st.json(debug)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "debug": debug,
                })
                save_chat_messages(chat_id, st.session_state.messages)

            except Exception as e:
                err = f"실행 오류: {e}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err,
                })
                save_chat_messages(chat_id, st.session_state.messages)
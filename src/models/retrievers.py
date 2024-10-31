from typing import List
import chromadb
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import ConfigurableField
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4

def setup_vector_store(docs: List[Document], embeddings: OpenAIEmbeddings):
    """Chromaベクトルストアのセットアップ"""
    try:
        persistent_client = chromadb.PersistentClient("./chroma_langchain_db")
        collection = persistent_client.get_collection("aggrement_docs_db")
        
        vector_store = Chroma(
            client=persistent_client,
            collection_name="aggrement_docs_db",
            embedding_function=embeddings,
        )
    except Exception:
        vector_store = Chroma(
            collection_name="aggrement_docs_db",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
    
    return vector_store

def create_ensemble_retriever(vector_store, preprocess_func, k: int = 3, score_threshold: float = 0.05):
    """Ensembleリトリーバーの作成"""
    chroma_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"k": k, "score_threshold": score_threshold}
    ).configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs_chroma",
            name="Search Kwargs",
            description="The search kwargs to use",
        )
    )

    bm25_retriever = BM25Retriever.from_documents(
        [Document(page_content=doc) for doc in vector_store.get()['documents']],
        preprocess_func=preprocess_func,
        k=k,
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], 
        weights=[0.5, 0.5]
    ) 
from dotenv import load_dotenv, find_dotenv
import re
from typing import List, Optional
from uuid import uuid4

import chromadb
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sudachipy import tokenizer
from sudachipy import dictionary


load_dotenv(find_dotenv())


pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Call_Center_Operation_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Consulting_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Content_Production_Service_Contract_(Request_Form).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Customer_Referral_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Draft_Editing_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Graphic_Design_Production_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/software_development_outsourcing_contracts.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Technical_Verification_(PoC)_Contract.pdf",
]



def load_pdfs_with_metadata(pdf_urls: List[str]) -> List[Document]:
    """
    複数のPDFドキュメントのURLからドキュメントを読み込み、メタデータを取得する。

    Args:
        pdf_urls: PDFドキュメントのURLのリスト。

    Returns:
        Documentのリスト
    """
    documents = []
    for url in pdf_urls:
        loader = PyPDFLoader(url)
        doc_text = ""
        for page in loader.lazy_load():
            add_content = re.sub(r'^\d+\s+', '', page.page_content.strip())
            doc_text += add_content
        # doc_metadata = get_metadata_with_text(llm, doc_text)
        metadata = {}
        metadata['agreement_title'] = doc_text.split('\n')[0].strip() # doc_metadata.agreement_title
        # metadata['contract_date'] = doc_metadata.contract_date
        # for ind, party in enumerate(doc_metadata.parties):
        #     metadata[party.abbreviation] = f'代表者:{party.name}\n会社名:{party.company_name}\n住所:{party.address}'
        
        documents.append(
            Document(
                page_content=doc_text,
                metadata=metadata
            )
        )
    return documents



llm =  ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


try:
    persistent_client = chromadb.PersistentClient("./chroma_langchain_db")
    collection = persistent_client.get_collection("aggrement_docs_db")

    vector_store = Chroma(
        client=persistent_client,
        collection_name="aggrement_docs_db",
        embedding_function=embeddings,
    )
    # print("既存のインデックスを読み込みました。")
except Exception as e:
    # print('Exception: ', e)
    docs = load_pdfs_with_metadata(pdf_file_urls)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200)

    chunk_docs = text_splitter.split_documents(docs)
    vector_store = Chroma(
        collection_name="aggrement_docs_db",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    uuids = [str(uuid4()) for _ in range(len(chunk_docs))]
    vector_store.add_documents(documents=chunk_docs, ids=uuids)
    # print("新規インデックスを作成しました")

# 日本語のトークン化関数の準備
def preprocess_func(text: str) -> List[str]:
    tokenizer_obj = dictionary.Dictionary(dict="core").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text ,mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))  # 重複削除
    return words

chroma_retriever = vector_store.as_retriever(
    # search_type="similarity",
    # search_kwargs={"k": 1},
    search_type="similarity_score_threshold", 
    search_kwargs={"k": 2, "score_threshold": 0.05}
).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs_chroma",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

# bm25
bm25_retriever = BM25Retriever.from_documents(
    [Document(page_content=doc) for doc in vector_store.get()['documents']],
    preprocess_func=preprocess_func,
    k=2,
)

# ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)



template = """Answer the question based only on the following context
# Context
{context}

# Constraints
- Return the answer in japanese only
- If there is no context, please respond with 'I don't understand.'
- If there is no information related to the context, please respond with 'There is no relevant information.'

# Question
{question}

# Answer
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-4o", temperature=0)



chain = (
    {"context": ensemble_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

question = "グラフィックデザイン制作業務委託契約について、受託者はいつまでに仕様書を作成して委託者の承諾を得る必要がありますか？"
answer = chain.invoke(question)
print(answer)
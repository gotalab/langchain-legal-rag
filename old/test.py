from dotenv import load_dotenv, find_dotenv
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import ConfigurableField
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)

import pdfplumber
from sudachipy import tokenizer
from sudachipy import dictionary

##############
# ドキュメントの準備
texts = [
    "後藤ひとりはギターが上手",
    "後藤ひとりの妹は後藤ふたり", 
    "虹夏の名字は伊地知", 
]

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]

pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Call_Center_Operation_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Consulting_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Content_Production_Service_Contract_(Request_Form).pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Customer_Referral_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Draft_Editing_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Graphic_Design_Production_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/software_development_outsourcing_contracts.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Technical_Verification_(PoC)_Contract.pdf",
]

#############

load_dotenv(find_dotenv())

# 日本語のトークン化関数の準備
def preprocess_func(text: str) -> List[str]:
    tokenizer_obj = dictionary.Dictionary(dict="core").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text ,mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))  # 重複削除
    return words

# ドキュメント読み込み
# PDFからテキスト抽出
def extract_text_from_pdf(pdf_path):
    """PDFからテキストを抽出"""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages])

# 文章をチャンクに分割
def split_text(text):
    """テキストをチャンクに分割するための関数"""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


# 形態素解析
def tokenize_text(text):
    tokenizer_obj = dictionary.Dictionary(dict="core").create()
    return [m.surface() for m in tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.C)]

url = "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",


# 埋め込みモデルの準備
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# VectorStoreの準備
vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
)

# Retrieverの準備
# chroma
chroma_retriever = vectorstore.as_retriever(
    # search_type="similarity",
    # search_kwargs={"k": 1},
    search_type="similarity_score_threshold", 
    search_kwargs={"k": 1, "score_threshold": 0.5}
).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs_chroma",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

# bm25
bm25_retriever = BM25Retriever.from_documents(
    docs, 
    preprocess_func=preprocess_func,
    k=1,
)

# ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)


# Retrieverの動作確認
# answer = chroma_retriever.invoke("ギターが得意なのは？")
# answer = ensemble_retriever.invoke("What are some movies about dinosaurs")


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

# What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated
print(chain.invoke("What are some movies about dinosaurs?"))
# print(ensemble_retriever.invoke("What are some movies about dinosaurs?"))
from dotenv import load_dotenv, find_dotenv
import re
from typing import List, Dict, Optional
from uuid import uuid4


from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


METADATA_EXTRACT_PROMPT_TEMPLATE = """以下のコンテキストに基づいて、次の形式でJSON出力を作成してください
情報がない場合は、Noneを入れてください

コンテキスト:
{context}
"""

class ContractParty(BaseModel):
    """契約の当事者を表します。"""
    abbreviation: str = Field(description="当事者の略称")
    name: str = Field(description="当事者の名前")
    company_name: Optional[str] = Field(default=None, description="当事者の会社名")
    address: Optional[str] = Field(default=None, description="当事者の住所")

# 契約全体の構造を表すモデル
class Contract(BaseModel):
    """契約書の構造を表します。"""
    agreement_title: str = Field(description="契約のタイトル")
    contract_date: str = Field(description="契約の日付")
    parties: List[ContractParty] = Field(description="契約の当事者たち")



def get_last_path_part(url: str) -> str:
    try:
        # URLの末尾にあるスラッシュを削除してからスラッシュで分割
        last_part = url.rstrip('/').split('/')[-1]
        
        # 拡張子が .pdf であれば除外
        if last_part.endswith('.pdf'):
            last_part = last_part[:-4]
        
        # アンダースコアをスペースに置換
        last_part = last_part.replace('_', ' ')
        
        return last_part
    except Exception as e:
        # エラーが発生したら元のURLを返す
        return url

def get_metadata_with_text(llm, text: str, prompt_template=METADATA_EXTRACT_PROMPT_TEMPLATE):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    structured_llm = llm.with_structured_output(Contract)
    chain = prompt | structured_llm

    i = 0
    while True:
        try:
            return chain.invoke({'context': text})
        except Exception:
            if i < 3:
                break
            i += 1

    return {}

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
        doc_metadata = get_metadata_with_text(llm, doc_text)
        metadata = {}
        metadata['agreement_title'] = doc_text.split('\n')[0].strip() # doc_metadata.agreement_title
        metadata['contract_date'] = doc_metadata.contract_date
        for ind, party in enumerate(doc_metadata.parties):
            metadata[party.abbreviation] = f'代表者:{party.name}\n会社名:{party.company_name}\n住所:{party.address}'
        
        documents.append(
            Document(
                page_content=doc_text,
                metadata=metadata
            )
        )
    return documents




from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

llm =  ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


docs = load_pdfs_with_metadata(pdf_file_urls)
print(len(docs))
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
# chunk_docs = text_splitter.split_documents(docs)

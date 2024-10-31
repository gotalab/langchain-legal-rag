import re
from typing import List
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs_with_metadata(pdf_urls: List[str]) -> List[Document]:
    """
    複数のPDFドキュメントのURLからドキュメントを読み込み、メタデータを取得する。
    """
    documents = []
    for url in pdf_urls:
        loader = PyPDFLoader(url)
        doc_text = ""
        for page in loader.lazy_load():
            add_content = re.sub(r'^\d+\s+', '', page.page_content.strip())
            doc_text += add_content
        metadata = {}
        metadata['agreement_title'] = doc_text.split('\n')[0].strip()
        
        documents.append(
            Document(
                page_content=doc_text,
                metadata=metadata
            )
        )
    return documents

def split_documents(docs: List[Document], chunk_size: int = 512, chunk_overlap: int = 200) -> List[Document]:
    """ドキュメントをチャンクに分割する"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs) 
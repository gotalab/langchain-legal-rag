import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import callbacks
from langchain_openai import OpenAIEmbeddings

from src.utils.config_loader import load_config
from src.data.loader import load_pdfs_with_metadata, split_documents
from src.data.preprocessor import preprocess_func
from src.models.retrievers import setup_vector_store, create_ensemble_retriever
from src.chains.rag_chain import RAGChain


# model = "gpt-4o-mini"
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

def rag_implementation(question: str) -> str:
    load_dotenv()
    
    # 設定の読み込み
    config_path = Path("config.yaml")
    config = load_config(config_path)
    
    # 埋め込みモデルの準備
    embeddings = OpenAIEmbeddings(
        model=config.embedding.model_name,
        **config.embedding.model_kwargs
    )
    
    
    # ドキュメントの準備
    docs = load_pdfs_with_metadata(pdf_file_urls)
    chunk_docs = split_documents(
        docs, 
        chunk_size=config.retriever.chunk_size,
        chunk_overlap=config.retriever.chunk_overlap
    )
    
    # ベクトルストアとリトリーバーの準備
    vector_store = setup_vector_store(chunk_docs, embeddings)
    ensemble_retriever = create_ensemble_retriever(
        vector_store, 
        preprocess_func,
        k=config.retriever.k,
        score_threshold=config.retriever.score_threshold
    )
    
    # RAGチェーンの作成と実行
    chain = RAGChain(ensemble_retriever, config).create()
    answer = chain.invoke(question)
    
    return str(answer)

def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        run_id = cb.traced_runs[0].id

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output))

if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)

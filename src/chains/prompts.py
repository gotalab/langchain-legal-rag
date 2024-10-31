from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate

class PromptTemplates(TypedDict):
    rag: str
    metadata_extraction: str
    error: str

# RAG用のプロンプトテンプレート
RAG_PROMPT_TEMPLATE = """Answer the question based only on the following context

# Context
{context}

# Constraints
- Return the answer in japanese only
- Please answer concisely and without excess or deficiency.
- If there is no information related to the context, please respond with 'There is no relevant information.'

# Question
{question}

# Answer
"""

# メタデータ抽出用のプロンプトテンプレート
METADATA_EXTRACT_PROMPT_TEMPLATE = """
契約書の内容から以下の情報を抽出してください：

# Context
{context}

# Required Information
- 契約書のタイトル
- 契約日
- 契約当事者（各当事者について）:
  - 略称
  - 名前
  - 会社名（ある場合）
  - 住所（ある場合）

# Output Format
契約書の構造を正確に解析し、JSON形式で出力してください。
"""

# エラーメッセージ用のテンプレート
ERROR_PROMPT_TEMPLATE = """
申し訳ありませんが、エラーが発生しました。

エラーの内容：
{error_message}

以下のいずれかの対応をお願いします：
1. 質問の内容を変更する
2. システム管理者に連絡する
"""

class PromptBuilder:
    """プロンプトテンプレートを管理するクラス"""
    
    @staticmethod
    def create_rag_prompt() -> ChatPromptTemplate:
        """RAG用のプロンプトテンプレートを作成"""
        return ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    @staticmethod
    def create_metadata_prompt() -> ChatPromptTemplate:
        """メタデータ抽出用のプロンプトテンプレートを作成"""
        return ChatPromptTemplate.from_template(METADATA_EXTRACT_PROMPT_TEMPLATE)
    
    @staticmethod
    def create_error_prompt() -> ChatPromptTemplate:
        """エラーメッセージ用のプロンプトテンプレートを作成"""
        return ChatPromptTemplate.from_template(ERROR_PROMPT_TEMPLATE)

# プロンプトテンプレートの辞書
TEMPLATES: PromptTemplates = {
    "rag": RAG_PROMPT_TEMPLATE,
    "metadata_extraction": METADATA_EXTRACT_PROMPT_TEMPLATE,
    "error": ERROR_PROMPT_TEMPLATE
} 
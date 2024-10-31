from typing import Literal, Dict
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="LLMプロバイダー"
    )
    model_name: str = Field(
        default="gpt-4o",
        description="モデル名"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="生成時の温度パラメータ"
    )
    model_kwargs: Dict = Field(
        default_factory=dict,
        description="その他のモデルパラメータ"
    )

class EmbeddingConfig(BaseModel):
    provider: Literal["openai"] = Field(
        default="openai",
        description="埋め込みモデルプロバイダー"
    )
    model_name: str = Field(
        default="text-embedding-3-large",
        description="埋め込みモデル名"
    )
    model_kwargs: Dict = Field(
        default_factory=dict,
        description="その他の埋め込みモデルパラメータ"
    )

class RetrieverConfig(BaseModel):
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="チャンクサイズ"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="チャンクオーバーラップ"
    )
    k: int = Field(
        default=3,
        gt=0,
        description="取得するドキュメント数"
    )
    score_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="類似度スコアの閾値"
    )

class RAGConfig(BaseModel):
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLMの設定"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="埋め込みモデルの設定"
    )
    retriever: RetrieverConfig = Field(
        default_factory=RetrieverConfig,
        description="リトリーバーの設定"
    ) 
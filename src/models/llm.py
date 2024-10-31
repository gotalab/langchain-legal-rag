from typing import Protocol
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from src.core.config import LLMConfig

class LLMFactory(Protocol):
    """LLMファクトリープロトコル"""
    def create_llm(self, config: LLMConfig) -> BaseLanguageModel:
        ...

class OpenAIFactory:
    def create_llm(self, config: LLMConfig) -> ChatOpenAI:
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            **config.model_kwargs
        )

class AnthropicFactory:
    def create_llm(self, config: LLMConfig) -> ChatAnthropic:
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            **config.model_kwargs
        )

class LLMFactoryBuilder:
    _factories = {
        "openai": OpenAIFactory(),
        "anthropic": AnthropicFactory(),
    }

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLanguageModel:
        factory = cls._factories.get(config.provider)
        if not factory:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        return factory.create_llm(config) 
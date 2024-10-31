from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from .prompts import RAG_PROMPT_TEMPLATE

class RAGChain:
    def __init__(self, retriever, config):
        self.retriever = retriever
        self.config = config
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.model = ChatOpenAI(
            model=str(config.llm.model_name), 
            temperature=config.llm.temperature
        )
        
    def create(self):
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        ) 
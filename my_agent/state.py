from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import operator

class State(TypedDict):
    question: str
    category: str
    documents: list[Document]
    web_search: str
    answer: str
    messages: Annotated[list[BaseMessage], operator.add]
    relevance_score: float
    grounding_score: float
    completeness_score: float
    final_confidence_score: float
    retry_count: int
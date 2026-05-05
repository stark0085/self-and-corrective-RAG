import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from state import State

class RetrievalEvaluator(BaseModel):
    """
    A class to evaluate the relevance of documents retrieved from the vector store.
    """
    score: float = Field(description="Relevance score between 0 and 1, where 1 is highly relevant and 0 is not relevant.")

# Use the centralized llm with structured output
structured_llm = llm.with_structured_output(RetrievalEvaluator)

system_prompt = """Evaluate the document to determine its relevance to the user's question. 
Assign a score between 0 and 1. 
- 1.0: The document directly and fully answers the question or provides essential information.
- 0.5: The document is somewhat related but doesn't fully answer the question.
- 0.0: The document is completely irrelevant.
Only respond with the score.
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'User Question: \n\n {question} \n\n Retrieved Document: \n\n {document}')
])

chain = prompt | structured_llm

def document_evaluator(state: State):
    """
    Evaluate the relevance of the retrieved documents

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated documents, web_search flag, and relevance_score
    """

    question = state['question']
    documents = state['documents']

    web_search = 'no'
    filtered_docs = []
    total_score = 0

    if not documents:
        return {"question": question, "documents": [], "web_search": 'yes', "relevance_score": 0.0}

    for document in documents:
        response = chain.invoke({"question": question, "document": document.page_content})
        total_score += response.score

        if response.score >= 0.5:
            filtered_docs.append(document)

    relevance_score = total_score / len(documents)

    if relevance_score <= 0.5:
        web_search = 'yes'
        
    return {"question": question, "documents": filtered_docs, "web_search": web_search, "relevance_score": relevance_score}
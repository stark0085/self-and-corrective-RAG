import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from state import State

class RetrievalEvaluator(BaseModel):
    score: float = Field(description="Relevance score between 0 and 1")

structured_llm = llm.with_structured_output(RetrievalEvaluator)

system_prompt = """Evaluate the document to determine its relevance to the user's question. 
Assign a score between 0 and 1. 
- 1.0: Directly and fully answers.
- 0.5: Somewhat related.
- 0.0: Completely irrelevant.
Only respond with the score."""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'User Question: \n\n {question} \n\n Retrieved Document: \n\n {document}')
])

chain = prompt | structured_llm

def document_evaluator(state: State):
    """Evaluates retrieved document relevance."""
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
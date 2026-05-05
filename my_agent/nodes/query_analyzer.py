import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from state import State

class QueryEvaluator(BaseModel):
    category: str = Field(description="Question category: greeting, relevant, or irrelevant")

system_prompt = """You are an AI support representative of 'Fantix LLC' tasked with identifying the type of user query. 
Classify each query into one of the following categories:

1. **Greeting**: A friendly or polite greeting.  
2. **Relevant**: A query seeking information related to the given context.  
3. **Irrelevant**: Queries unrelated to Fantix LLC.

Instructions:  
- Always return 'greeting', 'relevant', or 'irrelevant'.  
- Clearly determine the category based on content.  
- Ask for clarification only if the query is ambiguous."""

structured_llm = llm.with_structured_output(QueryEvaluator)

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'User Question: \n\n {question}')
])

chain = prompt | structured_llm

def analyze_query(state: State):
    """Categorizes the incoming user query."""
    question = state['question']
    response = chain.invoke({"question": question})
    return {"question": question, "category": response.category}
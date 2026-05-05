import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm
from state import State

system_prompt = """Transform the user’s question into a concise, clear, and search-engine-friendly query that focuses\
on retrieving relevant and accurate information. Ensure the phrasing avoids ambiguity and includes keywords\
likely to yield the best search results."""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'User Question: {question}')
])

chain = prompt | llm | StrOutputParser()

def transform_query(state: State):
    question = state['question']
    documents = state['documents']
    response = chain.invoke({"question": question})
    
    return {"question": response, "documents": documents}
import os
from langchain_chroma import Chroma
from config import embeddings
from state import State

def retriever(state: State):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="fantix-llc"
    )
    
    db_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    question = state['question']
    documents = db_retriever.invoke(question)

    return {"documents": documents}

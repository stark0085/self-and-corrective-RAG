import os
from langchain_chroma import Chroma
from config import embeddings
from state import State

def retriever(state: State):
    """
    Retrieve documents based on the user's question with lazy-loading ChromaDB

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated documents
    """
    
    # Lazy-loading ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="fantix-llc"
    )
    
    db_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    question = state['question']
    documents = db_retriever.invoke(question)

    return {"documents": documents}


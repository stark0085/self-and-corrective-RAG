from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from state import State

def tavily_web_search(state: State):
    question = state['question']
    print("QUESTION: ", question)

    tavily_search = TavilySearchResults(k=3)
    search_results = tavily_search.invoke(question)

    docs = '\n\n'.join(item['content'] for item in search_results)
    urls = '\n\n'.join(item['url'] for item in search_results)

    doc = docs + urls
    document = Document(page_content=doc, metadata={})

    return {"question": question, "documents": [document]}
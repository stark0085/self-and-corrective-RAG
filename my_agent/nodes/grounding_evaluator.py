from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from state import State

class GroundingEvaluator(BaseModel):
    score: float = Field(description="Grounding score between 0 and 1")

structured_llm = llm.with_structured_output(GroundingEvaluator)

system_prompt = """You are an expert evaluator. Check if the provided answer is fully supported by the retrieved documents.
Every claim must be backed by the context. 
Assign a score between 0 and 1:
- 1.0: All claims perfectly supported.
- 0.5: Partially supported or contains hallucination.
- 0.0: Not grounded at all.
Only respond with the score."""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'Retrieved Documents: \n\n {documents} \n\n Answer: \n\n {answer}')
])

chain = prompt | structured_llm

def evaluate_grounding(state: State):
    """Evaluates how well the answer is grounded in the provided documents."""
    answer = state['answer']
    documents = state['documents']
    context = "\n\n".join(doc.page_content for doc in documents)
    response = chain.invoke({"documents": context, "answer": answer})
    return {"grounding_score": response.score}

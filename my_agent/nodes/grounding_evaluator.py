from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from state import State

class GroundingEvaluator(BaseModel):
    """
    Evaluate if the answer is grounded in the retrieved documents.
    """
    score: float = Field(description="Grounding score between 0 and 1, where 1 means all claims are supported.")

structured_llm = llm.with_structured_output(GroundingEvaluator)

system_prompt = """You are an expert evaluator. Your task is to check if the provided answer is fully supported by the retrieved documents.
Every claim in the answer must be backed by the context. 
Assign a score between 0 and 1:
- 1.0: All claims are perfectly supported by the documents.
- 0.5: Some claims are supported, but others are not or are based on external knowledge.
- 0.0: The answer is not grounded in the documents at all.
Only respond with the score.
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'Retrieved Documents: \n\n {documents} \n\n Answer: \n\n {answer}')
])

chain = prompt | structured_llm

def evaluate_grounding(state: State):
    """
    Evaluate if the answer is grounded in the retrieved documents

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated grounding_score
    """
    answer = state['answer']
    documents = state['documents']
    
    context = "\n\n".join(doc.page_content for doc in documents)
    
    response = chain.invoke({"documents": context, "answer": answer})
    
    return {"grounding_score": response.score}

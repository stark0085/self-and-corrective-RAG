from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from state import State

class CompletenessEvaluator(BaseModel):
    """
    Evaluate if the answer fully addresses the user's question.
    """
    score: float = Field(description="Completeness score between 0 and 1, where 1 means the question is fully addressed.")

structured_llm = llm.with_structured_output(CompletenessEvaluator)

system_prompt = """You are an expert evaluator. Your task is to check if the provided answer fully addresses the user's original question.
Assign a score between 0 and 1:
- 1.0: The answer completely and accurately addresses all parts of the question.
- 0.5: The answer addresses the question partially or is missing key details.
- 0.0: The answer does not address the question at all.
Only respond with the score.
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'Question: \n\n {question} \n\n Answer: \n\n {answer}')
])

chain = prompt | structured_llm

def evaluate_completeness(state: State):
    """
    Evaluate if the answer fully addresses the user's question

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated completeness_score
    """
    question = state['question']
    answer = state['answer']
    
    response = chain.invoke({"question": question, "answer": answer})
    
    return {"completeness_score": response.score}

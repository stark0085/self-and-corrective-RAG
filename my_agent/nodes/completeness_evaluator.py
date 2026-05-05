from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from state import State

class CompletenessEvaluator(BaseModel):
    score: float = Field(description="Completeness score between 0 and 1")

structured_llm = llm.with_structured_output(CompletenessEvaluator)

system_prompt = """You are an expert evaluator. Check if the provided answer fully addresses the user's question.
Assign a score between 0 and 1:
- 1.0: Completely and accurately addresses all parts.
- 0.5: Addresses partially or missing key details.
- 0.0: Does not address at all.
Only respond with the score."""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', 'Question: \n\n {question} \n\n Answer: \n\n {answer}')
])

chain = prompt | structured_llm

def evaluate_completeness(state: State):
    """Evaluates answer completeness."""
    question = state['question']
    answer = state['answer']
    response = chain.invoke({"question": question, "answer": answer})
    return {"completeness_score": response.score}

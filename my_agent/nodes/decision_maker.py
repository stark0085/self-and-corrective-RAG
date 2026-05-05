from state import State

def decide_to_go_next(state: State):
    """Router based on document relevance."""
    relevance_score = state.get('relevance_score', 0.0)
    if relevance_score < 0.6:
        return 'transform_query'
    return 'generator'

def decide_response_generator(state: State):
    """Router based on query category."""
    category = state['category']
    if category == 'greeting' or category == 'irrelevant':
        return 'llm'
    if category == 'relevant':
        return 'retriever'
    return 'llm'

def check_generation_quality(state: State):
    """Evaluates if the generation needs revision."""
    grounding_score = state.get('grounding_score', 0.0)
    completeness_score = state.get('completeness_score', 0.0)
    retry_count = state.get('retry_count', 0)

    if (grounding_score < 0.7 or completeness_score < 0.7) and retry_count < 2:
        return 'revise_query'
    return 'confidence_scorer'

def increment_retry_count(state: State):
    return {"retry_count": state.get('retry_count', 0) + 1}
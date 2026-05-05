from state import State

def decide_to_go_next(state: State):
    """
    Decide whether to go to the next node based on the current state.
    Updated to use relevance_score.

    Args:
        state (State): Current state of the conversation
    Returns:
        str: Name of the next node to be executed
    """
    relevance_score = state.get('relevance_score', 0.0)

    if relevance_score < 0.6:
        return 'transform_query'
    
    return 'generator'


def decide_response_generator(state: State):
    """
    Decide which response generator to use based on the current state

    Args:
        state (State): Current state of the conversation
    Returns:
        str: Name of the response generator to be used
    """
    category = state['category']

    if category == 'greeting' or category == 'irrelevant':
        return 'llm'
    
    if category == 'relevant':
        return 'retriever'
    
    return 'llm'

def check_generation_quality(state: State):
    """
    Conditional edge to check the quality of the generated answer.
    
    Args:
        state (State): Current state of the conversation
    Returns:
        str: Next node to route to
    """
    grounding_score = state.get('grounding_score', 0.0)
    completeness_score = state.get('completeness_score', 0.0)
    retry_count = state.get('retry_count', 0)

    if (grounding_score < 0.7 or completeness_score < 0.7) and retry_count < 2:
        return 'revise_query'
    
    return 'confidence_scorer'

def increment_retry_count(state: State):
    """
    Node to increment the retry count in the state.
    
    Args:
        state (State): Current state of the conversation
    Returns:
        dict: Updated retry_count
    """
    return {"retry_count": state.get('retry_count', 0) + 1}
from state import State

def calculate_confidence(state: State):
    """
    Calculate a weighted average of the relevance, grounding, and completeness scores.

    Args:
        state (State): Current state of the conversation
    Returns:
        dict: State with updated final_confidence_score
    """
    relevance = state.get('relevance_score', 0.0)
    grounding = state.get('grounding_score', 0.0)
    completeness = state.get('completeness_score', 0.0)
    
    # Simple average for now, could be weighted if needed
    final_score = (relevance + grounding + completeness) / 3.0
    
    return {"final_confidence_score": final_score}

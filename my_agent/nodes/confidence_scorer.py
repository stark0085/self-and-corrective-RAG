from state import State

def calculate_confidence(state: State):
    """Calculates a weighted average of all evaluation scores."""
    relevance = state.get('relevance_score', 0.0)
    grounding = state.get('grounding_score', 0.0)
    completeness = state.get('completeness_score', 0.0)
    
    final_score = (relevance + grounding + completeness) / 3.0
    
    return {"final_confidence_score": final_score}

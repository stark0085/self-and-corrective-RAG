from langgraph.graph import StateGraph, START, END

from state import State
from nodes.retriever import retriever
from nodes.generator import generator
from nodes.evaluator import document_evaluator
from nodes.decision_maker import (
    decide_to_go_next, 
    decide_response_generator, 
    check_generation_quality,
    increment_retry_count
)
from nodes.query_transformer import transform_query
from nodes.web_search import tavily_web_search
from nodes.query_analyzer import analyze_query
from nodes.llm import customer_support
from nodes.grounding_evaluator import evaluate_grounding
from nodes.completeness_evaluator import evaluate_completeness
from nodes.confidence_scorer import calculate_confidence

graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node('query_analyzer', analyze_query)
graph_builder.add_node('chatbot', customer_support)
graph_builder.add_node('retriever', retriever)
graph_builder.add_node('generator', generator)
graph_builder.add_node('evaluator', document_evaluator)
graph_builder.add_node('query_transformer', transform_query)
graph_builder.add_node('search_web', tavily_web_search)
graph_builder.add_node('grounding_evaluator', evaluate_grounding)
graph_builder.add_node('completeness_evaluator', evaluate_completeness)
graph_builder.add_node('confidence_scorer', calculate_confidence)
graph_builder.add_node('revise_query', increment_retry_count)

# Edges
graph_builder.add_edge(START, 'query_analyzer')

graph_builder.add_conditional_edges(
    'query_analyzer',
    decide_response_generator,
    {
        "llm": 'chatbot',
        "retriever": 'retriever',
    }
)

graph_builder.add_edge('chatbot', END)

graph_builder.add_edge('retriever', 'evaluator')

graph_builder.add_conditional_edges(
    'evaluator',
    decide_to_go_next,
    {
        "transform_query": 'query_transformer',
        "generator": 'generator',
    }
)

graph_builder.add_edge('query_transformer', 'search_web')
graph_builder.add_edge('search_web', 'generator')

# Post-generation evaluation flow
graph_builder.add_edge('generator', 'grounding_evaluator')
graph_builder.add_edge('grounding_evaluator', 'completeness_evaluator')

graph_builder.add_conditional_edges(
    'completeness_evaluator',
    check_generation_quality,
    {
        "revise_query": 'revise_query',
        "confidence_scorer": 'confidence_scorer',
    }
)

# Retry logic: go back to query transformation for a better search/generation
graph_builder.add_edge('revise_query', 'query_transformer')

# Final score and end
graph_builder.add_edge('confidence_scorer', END)

graph = graph_builder.compile()
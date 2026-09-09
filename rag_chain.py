"""
Ties retrieval + LLM together.
"""
from src.retriever import RAGRetriever


PROMPT_TEMPLATE = """Use the following context to answer the question.
If the answer isn't in the context, say you don't know instead of guessing.

Context:
{context}

Question: {query}

Answer:"""


def _build_sources(results):
    return [
        {
            "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:300] + "...",
        }
        for doc in results
    ]


def rag_advanced(query, retriever: RAGRetriever, llm, top_k=5, min_score=0.2, return_context=False):
    """Non-streaming version — same behavior as your notebook function."""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {"answer": "No relevant context found", "sources": [], "confidence": 0.0, "context": ""}

    context = "\n\n".join([doc["content"] for doc in results])
    sources = _build_sources(results)
    confidence = max(doc["similarity_score"] for doc in results)

    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    response = llm.invoke([prompt])

    output = {"answer": response.content, "sources": sources, "confidence": confidence}
    if return_context:
        output["context"] = context
    return output


def get_rag_stream(query, retriever: RAGRetriever, llm, top_k=5, min_score=0.2):
    """Streaming version for the chat UI.

    Returns (token_generator, sources, confidence) so the caller can
    st.write_stream(token_generator) for the live-typing effect, then
    render sources/confidence separately once the stream finishes.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        def empty_gen():
            yield "I couldn't find anything relevant to that question in the uploaded documents. Try rephrasing, or upload a document that covers this topic."
        return empty_gen(), [], 0.0

    context = "\n\n".join([doc["content"] for doc in results])
    sources = _build_sources(results)
    confidence = max(doc["similarity_score"] for doc in results)

    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    def token_gen():
        for chunk in llm.stream([prompt]):
            if chunk.content:
                yield chunk.content

    return token_gen(), sources, confidence
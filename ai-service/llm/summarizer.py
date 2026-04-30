def summarize_chunk(chunk: str):
    # mock
    return {
        "summary": chunk[:200],
        "key_points": ["Point A", "Point B"],
    }

def summarize_document(chunk_summaries: list):
    combined = "\n".join([c["summary"] for c in chunk_summaries])

    # mock
    return {
        "summary": combined[:500],
        "key_ideas": ["Idea 1", "Idea 2"],
        "contributions": ["Contribution 1"]
    }
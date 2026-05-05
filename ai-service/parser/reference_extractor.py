from llm.summarizer import extract_references_llm

def extract_references(text: str):
    for keyword in ["References", "REFERENCES", "bibliography"]:
        if keyword in text:
            ref_text = text.split(keyword)[-1][:6000]
            
            return extract_references_llm(ref_text)
    
    return []

import re

def split_long_sentence(sentence: str, max_len: int) -> list[str]:
    return [
        sentence[i:i + max_len].strip()
        for i in range(0, len(sentence), max_len)
        if sentence[i:i + max_len].strip()
    ]

def chunk_text(text: str, max_len: int = 2500, min_len: int = 800) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(s) > max_len:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_long_sentence(s, max_len))
            continue

        if len(current) + len(s) + 1 <= max_len:
            current += s + " "
        else:
            if current and len(current) >= min_len:
                chunks.append(current.strip())
                current = s + " "
            else:
                current += s + " "
        
    if current:
        chunks.append(current.strip())
        
    return chunks

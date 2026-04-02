import re
import unicodedata

def remove_accents(input_str: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_for_comparison(text: str) -> str:
    if not text:
        return ""
    # Remove accents
    text = remove_accents(text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Compress whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

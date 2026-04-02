import re

# We use re.MULTILINE to allow ^ and $ to match the start/end of each line in the text
ARTICLE_RE = re.compile(r"^(Άρθρο|ΑΡΘΡΟ)\s+(\d+)([Α-ΩA-Z]?)\s*$", re.IGNORECASE | re.MULTILINE)
PARAGRAPH_START_RE = re.compile(r"^\d+\.\s+")
PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
CHAPTER_RE = re.compile(r"^\s*(ΚΕΦΑΛΑΙΟ|ΜΕΡΟΣ|ΤΜΗΜΑ)\b", re.IGNORECASE)
ENTRY_INTO_FORCE_TITLE_RE = re.compile(r"^\s*(Έναρξη\s+ισχύος|ΕΝΑΡΞΗ\s+ΙΣΧΥΟΣ)\s*$", re.IGNORECASE)

SIGNATURE_START_RE = re.compile(r"^\s*Αθήνα,\s+\d{1,2}\s+.+?\s+\d{4}\s*$", re.IGNORECASE | re.MULTILINE)
MINISTERS_HEADER_RE = re.compile(r"^\s*ΟΙ\s+ΥΠΟΥΡΓΟΙ\s*$", re.IGNORECASE | re.MULTILINE)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\ufeff", "")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_noise_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if PAGE_NUMBER_RE.match(s):
        return True
    if re.match(r"^[\-\–\—]?\s*\d+\s*[\-\–\—]?$", s):
        return True
    return False

def clean_noise_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = [line for line in lines if not is_noise_line(line)]
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_trailing_signature_block(text: str) -> str:
    if not text:
        return ""
    
    candidates = []
    
    date_match = SIGNATURE_START_RE.search(text)
    if date_match:
        candidates.append(date_match.start())
        
    ministers_match = MINISTERS_HEADER_RE.search(text)
    if ministers_match:
        candidates.append(ministers_match.start())
        
    if not candidates:
        return text.strip()
        
    cut_index = min(candidates)
    return text[:cut_index].strip()

def remove_trailing_chapters(text: str) -> str:
    if not text:
        return ""
    
    match = re.search(r"(?im)^\s*(ΚΕΦΑΛΑΙΟ|ΜΕΡΟΣ|ΤΜΗΜΑ)\b.*", text)
    
    if match:
        return text[:match.start()].strip()
    
    return text.strip()

def cut_before_first_article(text: str) -> str:
    text = normalize_text(text)
    text = clean_noise_lines(text)
    
    article1_re = re.compile(r"^(Άρθρο|ΑΡΘΡΟ)\s+1[Α-ΩA-Z]?\s*$", re.IGNORECASE | re.MULTILINE)
    article1 = article1_re.search(text)
    if article1:
        return text[article1.start():].strip()
        
    first_article_re = re.compile(r"^(Άρθρο|ΑΡΘΡΟ)\s+\d+[Α-ΩA-Z]?\s*$", re.IGNORECASE | re.MULTILINE)
    first_article = first_article_re.search(text)
    if first_article:
        return text[first_article.start():].strip()
        
    return text.strip()

def extract_title_and_body(lines: list[str]) -> tuple[str, str]:
    if len(lines) < 2:
        return "", ""
        
    title_lines = []
    body_lines = []
    found_boundary = False
    
    for i in range(1, len(lines)):
        line = lines[i].strip()
        
        if line == "":
            found_boundary = True
            body_lines = lines[i+1:]
            break
            
        if PARAGRAPH_START_RE.match(line) or CHAPTER_RE.match(line):
            found_boundary = True
            body_lines = lines[i:]
            break
            
        title_lines.append(line)
        
    if not found_boundary:
        # Fallback heuristic: Try to find where the title logically ends and body begins
        continuation_words = {"και", "ή", "ο", "η", "το", "του", "της", "των", "τον", "την", "τους", "τις", "τα", "σε", "με", "για", "από", "προς"}
        
        title_end_idx = 0
        for i in range(1, len(title_lines)):
            prev_line = title_lines[i-1].strip()
            curr_line = title_lines[i].strip()
            
            if not prev_line or not curr_line:
                continue
                
            # If current line starts with lowercase, it's definitely a continuation
            if curr_line[0].islower():
                title_end_idx = i
                continue
                
            # If previous line ends with a dash, comma, or continuation word, it's a continuation
            last_word = prev_line.split()[-1].lower() if prev_line.split() else ""
            if prev_line.endswith(("-", ",", "—", "–")) or last_word in continuation_words:
                title_end_idx = i
                continue
                
            # If current line starts with a Capital letter (or digit), we need to check if it's 
            # actually the body.
            if curr_line[0].isupper() or curr_line[0].isdigit():
                # Check if the previous line ended with a year/number (e.g. "2023")
                # and if the current line does not look like a numbered list item.
                # In Greek laws, body often starts with "Στο άρθρο...", "Η παρ...", "«1. ".
                # If prev line ended with a year, it's very likely the title is over, 
                # UNLESS it's a very specific case.
                # To be safe, if we reach here, we assume it's the body!
                break
                
            title_end_idx = i
            
        title = " ".join(title_lines[:title_end_idx + 1]).strip()
        body = "\n".join([b.strip() for b in title_lines[title_end_idx + 1:] if b.strip()]).strip()
        return title, body

    title = " ".join(title_lines).strip()
    body = "\n".join([b.strip() for b in body_lines if b.strip()]).strip()
    return title, body

def is_entry_into_force_title(title: str) -> bool:
    return bool(ENTRY_INTO_FORCE_TITLE_RE.match((title or "").strip()))

def get_article_matches(text: str) -> list[dict]:
    matches = []
    for match in ARTICLE_RE.finditer(text):
        matches.append({
            "index": match.start(),
            "number": match.group(2),
            "suffix": match.group(3) or ""
        })
    return matches

def split_top_level_articles(text: str) -> list[dict]:
    text = cut_before_first_article(text)
    matches = get_article_matches(text)
    articles = []
    
    if not matches:
        return []
        
    for i in range(len(matches)):
        match = matches[i]
        start = match["index"]
        end = matches[i + 1]["index"] if i + 1 < len(matches) else len(text)
        
        block = text[start:end].strip()
        block = clean_noise_lines(block)
        block = remove_trailing_signature_block(block)
        block = remove_trailing_chapters(block)
        
        if not block:
            continue
            
        raw_lines = [line.strip() for line in block.split("\n")]
        while raw_lines and not raw_lines[0]:
            raw_lines.pop(0)
        while raw_lines and not raw_lines[-1]:
            raw_lines.pop()
            
        if not raw_lines:
            continue
            
        header = raw_lines[0]
        number = f"{match['number']}{match['suffix']}"
        
        title, body = extract_title_and_body(raw_lines)
        
        article = {
            "article_number": number,
            "header": header,
            "title": title,
            "body": body,
        }
        articles.append(article)
        
        if is_entry_into_force_title(title):
            break
            
    return articles

def article_sort_key(article: dict) -> tuple[int, str]:
    number_str = article.get("article_number", "")
    m = re.match(r"^(\d+)([Α-ΩA-Z]?)$", number_str)
    if not m:
        return (999999, "")
    return (int(m.group(1)), m.group(2) or "")

def dedupe_by_longest(articles: list[dict]) -> list[dict]:
    best = {}
    for article in articles:
        key = article["article_number"]
        new_len = len(article.get("body", ""))
        old_len = len(best.get(key, {}).get("body", ""))
        
        if key not in best or new_len > old_len:
            best[key] = article
            
    sorted_articles = sorted(best.values(), key=article_sort_key)
    return sorted_articles

def extract_and_split_documents(documents: list) -> list[dict]:
    """
    Extracts text from a list of Langchain Document objects, 
    then splits and structures the text into top-level articles.
    """
    text = "\n".join(doc.page_content for doc in documents)
    articles = split_top_level_articles(text)
    return dedupe_by_longest(articles)

import re

PARAGRAPH_START_RE = re.compile(r"^\d+\.\s+")
CHAPTER_RE = re.compile(r"^\s*(ΚΕΦΑΛΑΙΟ|ΜΕΡΟΣ|ΤΜΗΜΑ)\b", re.IGNORECASE)

def extract_title_and_body(lines: list[str]) -> tuple[str, str]:
    if len(lines) < 2:
        return "", ""
        
    title_lines = []
    body_lines = []
    found_boundary = False
    
    # First, let's collect the title. It starts at lines[1] (lines[0] is the header)
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
            
        # If the line starts with a capital letter and the previous line ended with a period,
        # this might be the body! But titles don't end with periods usually.
        # Let's check if we already have at least 1 title line, and the current line
        # starts with a capital letter, AND it looks like a new sentence.
        # This is risky. 
        # A simpler heuristic: if line starts with lowercase, it's continuation.
        # If it starts with uppercase, is it a new sentence or a Name?
        
        title_lines.append(line)

    if not found_boundary:
        # Fallback: how do we split title_lines if there was no blank line?
        # Let's say the title continues as long as the next line starts with lowercase,
        # or the current line doesn't end with a period.
        # Actually, let's just find the first sentence that ends with a period, 
        # or the first line that is long.
        pass

    # For testing, let's just print
    return title_lines

lines_no_number = [
    "Άρθρο 61Α",
    "Προσκόμιση γραμματίου προείσπραξης σε πράξεις εκδοθείσες από δικηγόρο ή",
    "συμβολαιογράφο - Αντικατάσταση του πίνακα του άρθρου 61Α του Κώδικα Δικηγόρων",
    "Αν ο δικηγόρος αποφασίσει να...",
    "Τότε γίνεται αυτό."
]

print(extract_title_and_body(lines_no_number))

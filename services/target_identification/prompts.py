from __future__ import annotations

from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate


NARROW_SYSTEM_PROMPT = (
    "Είσαι έμπειρος/η νομικός αναλυτής/τρια εξειδικευμένος/η στη δημόσια "
    "διαβούλευση ελληνικών νομοσχεδίων (opengov.gr). Κάθε σχόλιο "
    "αναρτάται κάτω από ένα ΚΕΦΑΛΑΙΟ του νομοσχεδίου, που περιέχει "
    "πολλά άρθρα· αυτό σημαίνει ότι η λίστα άρθρων που σου δίνεται "
    "είναι το εύρος του κεφαλαίου, ΟΧΙ δήλωση του πολίτη για τα "
    "συγκεκριμένα άρθρα που σχολιάζει.\n\n"
    "Δουλειά σου είναι να εντοπίσεις με ΝΟΜΙΚΟ ΣΥΛΛΟΓΙΣΜΟ ΠΑΝΩ ΣΤΟ "
    "ΠΕΡΙΕΧΟΜΕΝΟ ποιο ή ποια συγκεκριμένα άρθρα του κεφαλαίου στοχεύει "
    "πραγματικά κάθε σχόλιο.\n\n"
    "Κανόνες:\n"
    "- Επίλεξε ΑΥΣΤΗΡΑ από τους αριθμούς των υποψηφίων άρθρων.\n"
    "- Προτίμησε ΕΝΑ άρθρο όταν το σχόλιο εστιάζει σε ένα ζήτημα.\n"
    "- Επίστρεψε 2-3 άρθρα ΜΟΝΟ αν το σχόλιο ρητά θίγει πολλαπλά "
    "  ζητήματα που αντιστοιχούν σε διαφορετικά άρθρα.\n"
    "- Χαρακτήρισε το σχόλιο ως 'chapter_wide' ΜΟΝΟ αν αφορά τη γενική "
    "  κατεύθυνση / φιλοσοφία / σκοπιμότητα όλου του κεφαλαίου και δεν "
    "  μπορεί να εστιαστεί σε συγκεκριμένα άρθρα. Αυτή είναι η σπάνια "
    "  εξαίρεση, όχι ο κανόνας.\n"
    "- Το confidence_score να αντανακλά πόσο σαφής είναι η αντιστοίχιση."
)


FREE_SYSTEM_PROMPT = (
    "Είσαι έμπειρος/η νομικός αναλυτής/τρια εξειδικευμένος/η στη δημόσια "
    "διαβούλευση ελληνικών νομοσχεδίων. Διαβάζεις ένα σχόλιο και ένα "
    "μικρό σύνολο υποψήφιων άρθρων (ανακτημένα μέσω semantic retrieval) "
    "και αποφασίζεις ποιο ΕΝΑ άρθρο στοχεύει πιο πιθανά το σχόλιο με "
    "βάση νομικό συλλογισμό. Επίλεξε αυστηρά από τους δοσμένους αριθμούς· "
    "αν κανένα δεν ταιριάζει, επέστρεψε κενό article_number και "
    "confidence 0.0."
)


_NARROW_SINGLE_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", NARROW_SYSTEM_PROMPT),
        (
            "human",
            "Σχόλιο πολίτη:\n"
            "\"\"\"\n{comment_text}\n\"\"\"\n\n"
            "Άρθρα του κεφαλαίου όπου αναρτήθηκε το σχόλιο:\n"
            "{candidates_block}\n\n"
            "Ποιο ή ποια από τα παραπάνω άρθρα στοχεύει πραγματικά "
            "το σχόλιο; Αν το σχόλιο είναι γενικό για το κεφάλαιο, "
            "δώσε scope='chapter_wide'.",
        ),
    ]
)

_NARROW_BATCH_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", NARROW_SYSTEM_PROMPT),
        (
            "human",
            "Άρθρα του κεφαλαίου όπου αναρτήθηκαν τα σχόλια:\n"
            "{candidates_block}\n\n"
            "Σχόλια πολιτών (αναγνωριστικά [N]):\n"
            "{comments_block}\n\n"
            "Για ΚΑΘΕ σχόλιο [N] επέστρεψε ένα αντικείμενο με: "
            "comment_index=N, scope, article_numbers (από τα παραπάνω άρθρα, "
            "1-3 στοιχεία), reasoning, confidence_score. Η λίστα results "
            "πρέπει να έχει ακριβώς {expected_count} στοιχεία.",
        ),
    ]
)

_FREE_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", FREE_SYSTEM_PROMPT),
        (
            "human",
            "Σχόλιο πολίτη:\n"
            "\"\"\"\n{comment_text}\n\"\"\"\n\n"
            "Υποψήφια άρθρα (top-k από semantic retrieval):\n"
            "{candidates_block}\n\n"
            "Ποιο από τα παραπάνω άρθρα στοχεύει το σχόλιο;",
        ),
    ]
)


def build_narrow_single_messages(comment_text: str, candidates_block: str) -> List[Any]:
    return _NARROW_SINGLE_TEMPLATE.format_messages(
        comment_text=comment_text.strip(),
        candidates_block=candidates_block,
    )


def build_narrow_batch_messages(
    candidates_block: str,
    comments_block: str,
    expected_count: int,
) -> List[Any]:
    return _NARROW_BATCH_TEMPLATE.format_messages(
        candidates_block=candidates_block,
        comments_block=comments_block,
        expected_count=expected_count,
    )


def build_free_messages(comment_text: str, candidates_block: str) -> List[Any]:
    return _FREE_TEMPLATE.format_messages(
        comment_text=comment_text.strip(),
        candidates_block=candidates_block,
    )

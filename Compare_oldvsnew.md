# Compare and align Service

Το service πρέπει να έχει 3 βασικά στάδια:

## 1. Article Matching
Σκοπός: να βρει ποιο παλιό άρθρο αντιστοιχεί σε ποιο νέο.

Δεν κάνουμε σύγκριση 1 προς 1 με βάση τη σειρά, γιατί μπορεί:
- να έχει προστεθεί νέο άρθρο
- να έχει αφαιρεθεί άρθρο
- να έχει αλλάξει η αρίθμηση

Άρα για κάθε παλιό άρθρο υπολογίζουμε similarity με τα νέα και κάνουμε matching.

## 2. Article Diff
Αφού βρεθεί το σωστό pair old → new, υπολογίζουμε τι άλλαξε μέσα στο άρθρο.

Εδώ βρίσκουμε:
- added text
- removed text
- modified segments

## 3. Change Classification
Με βάση το diff αποφασίζουμε αν το άρθρο είναι:
- unchanged
- modified
- added
- removed
- renumbered

## Προτεινόμενη δομή σε Python

```text
services/
  normalizer.py
  scorer.py
  matcher.py
  differ.py
  significance.py
  pipeline.py
models/
  models.py

# Final Version — Compare Articles Node

## Στόχος

Το node παίρνει:

- `old_articles`
- `new_articles`

και επιστρέφει για κάθε άρθρο αν είναι:

- `unchanged`
- `modified`
- `renumbered`
- `renumbered_modified`
- `added` (Δεν υπάρχει στο old και υπάρχει στο new)
- `removed` (Υπάρχει μόνο στο old και όχι στο new)

---

# Βασική λογική

Η σύγκριση **δεν γίνεται 1 προς 1 με βάση τη θέση**.

Αντί γι’ αυτό:

1. καθαρίζουμε τα άρθρα
2. βρίσκουμε ποια νέα άρθρα είναι πιθανοί υποψήφιοι για κάθε παλιό
3. υπολογίζουμε score ομοιότητας
4. κάνουμε one-to-one matching
5. ταξινομούμε το αποτέλεσμα
6. μετά μόνο στα matched pairs κάνουμε diff

---

# Προτεινόμενη flow

old_articles + new_articles
   ↓
normalize articles
   ↓
tokenize articles
   ↓
compute TF-IDF vectors
   ↓
compute embeddings
   ↓
retrieve top-k candidates per old article
   ↓
score each candidate pair
   ↓
one-to-one matching
   ↓
mark unmatched old as removed
   ↓
mark unmatched new as added
   ↓
diff matched pairs
   ↓
classify final change type
   ↓
return results
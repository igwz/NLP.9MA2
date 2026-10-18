import spacy

nlp = spacy.load("en_core_web_sm")

# Task 1
sen = nlp("I like watching football matches. I do not know how to play it though")
play = sen[12]
print("Word:", play.text)
print("Coarse POS:", play.pos_)
print("Fine POS:", play.tag_)
print("Meaning:", spacy.explain(play.tag_))

print(f"\n{'Word':<14} {'POS':<10} {'Tag':<8} Meaning")
for word in sen:
    print(f"{word.text:<14} {word.pos_:<10} {word.tag_:<8} {spacy.explain(word.tag_)}")

# Task 2
examples = [
    ("Did you search on Google?", "google"),
    ("Let us google it.", "google"),
    ("The has placed the food on the table", "table"),
    ("She plans to table a motion for debate in Parliament", "table"),
]

for sentence, target in examples:
    doc = nlp(sentence)
    word = next(token for token in doc if token.text.lower() == target)
    print("Sentence:", sentence)
    print(f"{word.text}: {word.pos_}, {word.tag_}, {spacy.explain(word.tag_)}\n")

# Task 3
from spacy import displacy

pos_doc = nlp("I like to play football. I hated it in my childhood though")
displacy.render(pos_doc, style="dep", jupyter=True, options={"distance": 110})

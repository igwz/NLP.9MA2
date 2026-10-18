import spacy

nlp = spacy.load("en_core_web_sm")

# Task 1
sentence = input("Enter a sentence: ").strip()
print("Sentence:", sentence)
user_doc = nlp(sentence)

if not user_doc.ents:
    print("No named entities were identified.")
for entity in user_doc.ents:
    print("Entity:", entity.text)
    print("Label:", entity.label_)
    print("Meaning:", spacy.explain(entity.label_))
    print("Token positions:", entity.start, entity.end)
    print("Character positions:", entity.start_char, entity.end_char)
    print()

# Task 2
from collections import Counter
from spacy import displacy

paragraph = (
    "This agreement is made and entered into by and between ‘Abc & Co.’  "
    "and ‘Bcd LLC’ for term 1 year starting from April 1, 2020,  hereinafter "
    "collectively referred to as the Parties. Samuel P. Jackson in the place "
    "(New York) and on the date written below, with the following terms and conditions."
)
paragraph_nlp = spacy.load("en_core_web_sm")
before_doc = paragraph_nlp(paragraph)
print("Original entities:", [(ent.text, ent.label_) for ent in before_doc.ents])

# Check the named parties, person, location, and start date in this passage.
expected_entities = [
    ("Abc & Co.", "ORG"),
    ("Bcd LLC", "ORG"),
    ("Samuel P. Jackson", "PERSON"),
    ("New York", "GPE"),
    ("April 1, 2020", "DATE"),
]
existing_entities = {(ent.text, ent.label_) for ent in before_doc.ents}
missing_entities = [item for item in expected_entities if item not in existing_entities]
print("Missing or incorrectly labelled entities:", missing_entities)

if missing_entities:
    ruler = paragraph_nlp.add_pipe("entity_ruler", after="ner", config={"overwrite_ents": True})
    ruler.add_patterns([
        {"label": label, "pattern": name} for name, label in missing_entities
    ])

agreement_doc = paragraph_nlp(paragraph)
print("Updated entities:", [(ent.text, ent.label_) for ent in agreement_doc.ents])
entity_counts = Counter(ent.label_ for ent in agreement_doc.ents)
print("PERSON count:", entity_counts["PERSON"])
print("ORG count:", entity_counts["ORG"])
displacy.render(agreement_doc, style="ent", jupyter=True)

# Task 3
from pathlib import Path
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
from spacy.util import fix_random_seed

fix_random_seed(42)
training_text = (
    "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico. "
    "It is known for its Windows operating system and Office software products."
)
entity_labels = [
    ("Microsoft", "ORG"),
    ("Bill Gates", "PERSON"),
    ("Paul Allen", "PERSON"),
    ("Albuquerque", "GPE"),
    ("New Mexico", "GPE"),
    ("Windows", "PRODUCT"),
    ("Office", "PRODUCT"),
]
annotations = {"entities": [
    (training_text.index(name), training_text.index(name) + len(name), label)
    for name, label in entity_labels
]}

custom_nlp = spacy.blank("en")
ner = custom_nlp.add_pipe("ner")
for _, label in entity_labels:
    ner.add_label(label)

training_doc = custom_nlp.make_doc(training_text)
alignment = offsets_to_biluo_tags(training_doc, annotations["entities"])
if "-" in alignment:
    raise ValueError("An entity does not align with token boundaries.")
example = Example.from_dict(training_doc, annotations)
optimizer = custom_nlp.initialize(get_examples=lambda: [example])

training_losses = []
for iteration in range(20):
    losses = {}
    example = Example.from_dict(custom_nlp.make_doc(training_text), annotations)
    custom_nlp.update([example], sgd=optimizer, drop=0.1, losses=losses)
    training_losses.append(losses["ner"])
    print(f"Iteration {iteration + 1}/20 - NER loss: {losses['ner']:.4f}")

model_path = Path("custom_ner_model")
custom_nlp.to_disk(model_path)
loaded_nlp = spacy.load(model_path)

test_sentence = "Bill Gates and Paul Allen founded Microsoft."
custom_doc = loaded_nlp(test_sentence)
print("\nSentence:", test_sentence)
if not custom_doc.ents:
    print("No entities were identified by the custom model.")
for entity in custom_doc.ents:
    print(entity.text, entity.label_, spacy.explain(entity.label_))

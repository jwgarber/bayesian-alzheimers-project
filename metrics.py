import pandas as pd
import spacy

df = pd.read_csv("./transcripts.csv")

df = pd.read_csv("./transcripts.csv")

nlp = spacy.load("en_core_web_sm")

text = "The boy is reaching for cookies."

doc = nlp(text)

for token in doc:
    print(token.text, token.pos_)

from collections import Counter

def extract_pos_features(text):
    doc = nlp(text)
    
    pos_counts = Counter(token.pos_ for token in doc)

    total = len(doc)

    return {
        "nouns": pos_counts["NOUN"],
        "verbs": pos_counts["VERB"],
        "adjectives": pos_counts["ADJ"],
        "adverbs": pos_counts["ADV"],
        "pronouns": pos_counts["PRON"],
        "noun_ratio": pos_counts["NOUN"]/total,
        "verb_ratio": pos_counts["VERB"]/total
    }

features = df["transcript"].apply(extract_pos_features)

feature_df = pd.DataFrame(list(features))

df = pd.concat([df, feature_df], axis=1)
print(df)

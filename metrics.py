import pandas as pd
import numpy as np
import spacy
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Feature extraction
def extract_features(text):
    doc = nlp(str(text))
    
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    total_words = len(tokens)
    
    if total_words == 0:
        return {k: 0 for k in FEATURE_NAMES}
    
    pos_counts = Counter(t.pos_ for t in tokens)
    
    nouns = pos_counts["NOUN"]
    verbs = pos_counts["VERB"]
    pronouns = pos_counts["PRON"]
    adjectives = pos_counts["ADJ"]
    adverbs = pos_counts["ADV"]
    
    function_words = (
        pos_counts["PRON"] + pos_counts["DET"] + pos_counts["ADP"] +
        pos_counts["AUX"] + pos_counts["CCONJ"] + pos_counts["SCONJ"] +
        pos_counts["PART"]
    )
    content_words = nouns + verbs + adjectives + adverbs
    
    unique_words = len({t.lower_ for t in tokens})
    
    # New features
    sent_lengths = [len([t for t in sent if not t.is_punct]) for sent in doc.sents]
    avg_sentence_length = np.mean(sent_lengths) if sent_lengths else 0
    sentence_length_std = np.std(sent_lengths) if sent_lengths else 0
    
    fillers = {"um", "uh", "er", "ah"}
    filler_count = sum(1 for t in tokens if t.lower_ in fillers)
    filler_ratio = filler_count / total_words
    
    repetition_count = sum(
        1 for i in range(1, len(tokens))
        if tokens[i].lower_ == tokens[i-1].lower_
    )
    repetition_ratio = repetition_count / total_words
    
    avg_word_length = np.mean([len(t.text) for t in tokens])
    pos_diversity = len(set(t.pos_ for t in tokens)) / total_words
    
    def safe_div(a, b):
        return a / b if b != 0 else 0
    
    return {
        "noun_ratio": safe_div(nouns, total_words),
        "verb_ratio": safe_div(verbs, total_words),
        "pronoun_ratio": safe_div(pronouns, total_words),
        "adjective_ratio": safe_div(adjectives, total_words),
        "adverb_ratio": safe_div(adverbs, total_words),
        "noun_verb_ratio": safe_div(nouns, verbs),
        "pronoun_noun_ratio": safe_div(pronouns, nouns),
        "function_word_ratio": safe_div(function_words, total_words),
        "content_word_ratio": safe_div(content_words, total_words),
        "type_token_ratio": safe_div(unique_words, total_words),
        "avg_sentence_length": avg_sentence_length,
        "sentence_length_std": sentence_length_std,
        "filler_ratio": filler_ratio,
        "repetition_ratio": repetition_ratio,
        "avg_word_length": avg_word_length,
        "pos_diversity": pos_diversity
    }

# Feature names
FEATURE_NAMES = [
    "noun_ratio","verb_ratio","pronoun_ratio","adjective_ratio","adverb_ratio",
    "noun_verb_ratio","pronoun_noun_ratio","function_word_ratio","content_word_ratio",
    "type_token_ratio","avg_sentence_length","sentence_length_std",
    "filler_ratio","repetition_ratio","avg_word_length","pos_diversity"
]


# Dataset processing
def process_dataset(input_csv, output_csv):
    
    df = pd.read_csv(input_csv)
    
    feature_list = [extract_features(text) for text in df["transcript"]]
    
    feature_df = pd.DataFrame(feature_list)
    df = pd.concat([df, feature_df], axis=1)
    
    scaler = StandardScaler()
    df[FEATURE_NAMES] = scaler.fit_transform(df[FEATURE_NAMES])
    
    df.to_csv(output_csv, index=False)
    return df

# Logistic Regression + importance
def run_logistic_regression(df, label_column="ad"):
    
    X = df[FEATURE_NAMES].values
    y = df[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Model Performance ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Feature importance (coefficients)
    print("\n=== Feature Importance (Coefficients) ===")
    for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
        print(f"{name:25s}: {coef:.3f}")
    
    # Permutation importance (better)
    print("\n=== Permutation Importance ===")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    for i in result.importances_mean.argsort()[::-1]:
        print(f"{FEATURE_NAMES[i]:25s}: {result.importances_mean[i]:.3f}")
    
    return model

# MAIN
if __name__ == "__main__":
    
    df = process_dataset("./transcripts.csv", "processed_features.csv")
    
    print(df.head())
    
    model = run_logistic_regression(df)
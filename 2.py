import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np

nltk.download('punkt')


def read_txt_file(file_path):
    file_path = Path(file_path)
    text_content = file_path.read_text(encoding="utf-8").strip()
    return text_content


def custom_nltk_tokenizer(text):
    raw_tokens = word_tokenize(text.lower())
    filtered_tokens = [
        token for token in raw_tokens if token.isalpha() and len(token) >= 2]
    return filtered_tokens


tfidf_vectorizer = TfidfVectorizer(
    tokenizer=custom_nltk_tokenizer,
    stop_words='english',
    max_features=50
)


def compute_tfidf(corpus, document_titles):
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return feature_names, tfidf_matrix, document_titles


if __name__ == "__main__":
    document_paths = [
        Path("reading.txt"),
        Path("citylife.txt"),
        Path("aidevelopment.txt")
    ]
    document_titles = [
        "Importance of Reading",
        "Urban vs Rural Life",
        "Artificial Intelligence Development"
    ]

    text_corpus = []
    for file_path in document_paths:
        text_content = read_txt_file(file_path)
        text_corpus.append(text_content)

    feature_names, tfidf_matrix, doc_titles = compute_tfidf(
        text_corpus, document_titles)
    for idx, title in enumerate(doc_titles):
        tfidf_scores = tfidf_matrix[idx].toarray()[0].round(3)
        score_str = " | ".join(
            [f"{word}:{score}" for word, score in zip(feature_names, tfidf_scores)])
        print(f"{title:<40} | {score_str}")

    new_text = "I like reading, and reading makes me happy.Reading broadens people's horizons"
    new_vector = tfidf_vectorizer.transform([new_text])
    s = cosine_similarity(new_vector, tfidf_matrix)
    print(s)
    print(doc_titles[np.argmax(s)])
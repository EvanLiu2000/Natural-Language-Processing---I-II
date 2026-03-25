import math
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Define basic Russian stopwords (excluded from tokenization)
RUSSIAN_STOPWORDS = {'этот', 'на', 'с', 'и',
                     'еще', 'тоже', 'второй', 'третий', 'четвертый'}


def preprocess_text_russian(text: str) -> List[str]:
    # Convert text to lowercase
    text_lower = text.lower()
    # Split text into tokens (whitespace-separated)
    tokens = text_lower.split()
    # Remove empty strings and stopwords
    tokens_clean = [
        token for token in tokens if token and token not in RUSSIAN_STOPWORDS]
    return tokens_clean


def calculate_tf(tokens: List[str], use_relative: bool) -> Dict[str, float]:
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {}

    tf_dict = {}
    # Count raw occurrences of each term
    for token in tokens:
        tf_dict[token] = tf_dict.get(token, 0) + 1

    # Calculate relative frequency if required
    if use_relative:
        for token in tf_dict:
            tf_dict[token] = tf_dict[token] / total_tokens

    return tf_dict


def calculate_idf(doc_tokens_list: List[List[str]]) -> Dict[str, float]:
    total_docs = len(doc_tokens_list)
    doc_count = {}  # Track number of documents containing each term

    # Count document frequency for each term
    for doc_tokens in doc_tokens_list:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_count[token] = doc_count.get(token, 0) + 1

    # Calculate IDF using natural logarithm (matches Sklearn's default)
    idf_dict = {}
    for token, df in doc_count.items():
        idf_dict[token] = math.log((total_docs + 1) / (df + 1)) + 1

    return idf_dict


def calculate_tfidf_matrix(doc_tokens_list: List[List[str]]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # 1. Calculate TF for all documents (absolute frequency)
    all_tf = [calculate_tf(tokens, True) for tokens in doc_tokens_list]

    # 2. Calculate global IDF values
    idf_dict = calculate_idf(doc_tokens_list)

    # 3. Build vocabulary (all unique terms across documents)
    vocab = sorted({token for doc in doc_tokens_list for token in doc})

    # 4. Construct TF-IDF matrix
    tfidf_matrix = []
    for doc_tf in all_tf:
        # Calculate TF-IDF for each term in vocabulary
        doc_vector = [doc_tf.get(token, 0.0) *
                      idf_dict.get(token, 0.0) for token in vocab]
        tfidf_matrix.append(doc_vector)

    # 5. Convert to DataFrame with meaningful indices
    doc_indices = [f"document{i+1}" for i in range(len(doc_tokens_list))]
    df_tfidf = pd.DataFrame(tfidf_matrix, index=doc_indices, columns=vocab)

    return df_tfidf, doc_indices, vocab


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    # Calculate L2 norm for each document vector
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero (set norm=1 for empty vectors)
    norms[norms == 0] = 1.0
    # Normalize matrix by dividing with norms
    normalized_matrix = matrix / norms
    return normalized_matrix


def compare_with_sklearn(docs: List[str]):
    # Preprocess all documents
    doc_tokens_list = [preprocess_text_russian(doc) for doc in docs]

    # Custom TF-IDF (without normalization)
    df_manual_no_norm, doc_indices, vocab = calculate_tfidf_matrix(
        doc_tokens_list)

    # Sklearn TF-IDF (without normalization)
    tfidf_sklearn_no_norm = TfidfVectorizer(
        tokenizer=preprocess_text_russian,  # Use same preprocessing
        lowercase=False,  # Preprocessing already lowercased text
        norm=None,        # No normalization
        smooth_idf=True,  # Match custom IDF formula
        sublinear_tf=False  # No sublinear TF scaling
    )
    # Fit and transform documents to TF-IDF matrix
    X_sklearn_no_norm = tfidf_sklearn_no_norm.fit_transform(docs).toarray()

    # Convert to DataFrame for easy comparison
    df_sklearn_no_norm = pd.DataFrame(
        X_sklearn_no_norm,
        index=doc_indices,
        columns=tfidf_sklearn_no_norm.get_feature_names_out()
    )

    # Custom TF-IDF (with L2 normalization)
    manual_matrix_no_norm = df_manual_no_norm.values
    manual_matrix_l2 = l2_normalize(manual_matrix_no_norm)
    df_manual_l2 = pd.DataFrame(
        manual_matrix_l2, index=doc_indices, columns=vocab)

    # Sklearn TF-IDF (with L2 normalization)
    tfidf_sklearn_l2 = TfidfVectorizer(
        tokenizer=preprocess_text_russian,
        lowercase=False,
        norm='l2',         # L2 normalization
        smooth_idf=True,
        sublinear_tf=False
    )
    X_sklearn_l2 = tfidf_sklearn_l2.fit_transform(docs).toarray()
    df_sklearn_l2 = pd.DataFrame(
        X_sklearn_l2,
        index=doc_indices,
        columns=tfidf_sklearn_l2.get_feature_names_out()
    )

    # Print comparison results
    print("=== 1. TF-IDF Comparison (No Normalization) - Custom vs Sklearn ===")
    # Align vocabulary for fair comparison (intersection of terms)
    common_vocab = sorted(set(vocab) & set(
        tfidf_sklearn_no_norm.get_feature_names_out()))
    print("\nCustom Implementation:")
    print(df_manual_no_norm[common_vocab].round(4).head())
    print("\nSklearn Implementation:")
    print(df_sklearn_no_norm[common_vocab].round(4).head())

    print("\n=== 2. TF-IDF Comparison (L2 Normalization) - Custom vs Sklearn ===")
    common_vocab_l2 = sorted(set(vocab) & set(
        tfidf_sklearn_l2.get_feature_names_out()))
    print("\nCustom Implementation:")
    print(df_manual_l2[common_vocab_l2].round(4).head())
    print("\nSklearn Implementation:")
    print(df_sklearn_l2[common_vocab_l2].round(4).head())

    # Validate numerical consistency
    # Check differences for no normalization
    manual_vals = df_manual_no_norm[common_vocab].values
    sklearn_vals = df_sklearn_no_norm[common_vocab].values
    diff_no_norm = np.abs(manual_vals - sklearn_vals).max()
    print(f"\n=== Numerical Differences ===")
    print(f"Max difference (No Normalization): {diff_no_norm:.6f}")

    # Check differences for L2 normalization
    manual_l2_vals = df_manual_l2[common_vocab_l2].values
    sklearn_l2_vals = df_sklearn_l2[common_vocab_l2].values
    diff_l2 = np.abs(manual_l2_vals - sklearn_l2_vals).max()
    print(f"Max difference (L2 Normalization): {diff_l2:.6f}")

    return {
        "manual_no_norm": df_manual_no_norm,
        "sklearn_no_norm": df_sklearn_no_norm,
        "manual_l2": df_manual_l2,
        "sklearn_l2": df_sklearn_l2
    }


if __name__ == "__main__":
    docs = [
        "этот текст на русском языке содержит несколько слов",
        "второй документ тоже на русском языке с некоторыми словами",
        "третий документ содержит уникальное слово редкость и еще слова",
        "четвертый документ повторяет слова документ и содержит"
    ]
    results = compare_with_sklearn(docs)

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Download NLTK Dependencies
import nltk
nltk.download('punkt')
nltk.download('stopwords')

text = """
The Magic of Travel

Travel is more than just visiting new places; it is an awakening of the soul. Stepping off a plane into an unfamiliar world, every sense is heightened—the aroma of strange spices, the melody of a foreign language, and the warmth of a different sun.

Wandering through ancient streets or quiet villages, we shed our routines and become explorers. We learn that kindness needs no translation and that adventure hides around every corner. Travel doesn't just show us the world; it shows us who we are when everything is new. In the end, we return home not with souvenirs, but with a heart full of stories and a spirit forever changed.
"""


def preprocess_text_modified(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenization
    tokens = word_tokenize(text)
    # 4. Filter stop words and empty characters
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        token for token in tokens if token not in stop_words and token.strip()]

    window_size = 5
    step = 2
    sentences = []
    for i in range(0, len(filtered_tokens) - window_size + 1, step):
        sentences.append(filtered_tokens[i:i + window_size])

    return sentences


processed_sentences = preprocess_text_modified(text)
print(f"Number of sentences (windows): {len(processed_sentences)}")
print(
    f"Example window: {processed_sentences[0] if processed_sentences else 'None'}")

# Train Word2Vec Model
model = Word2Vec(
    sentences=processed_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    epochs=200
)

# Query words similar to 'travel'
st = model.wv.most_similar("travel", topn=10)
for word, s in st:
    print(f"   {word} → Similarity: {s:.4f}")

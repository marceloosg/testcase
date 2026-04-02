import gensim.downloader as api
import numpy as np

print("Loading GloVe model (glove-wiki-gigaword-100)...")
model = api.load("glove-wiki-gigaword-100")
print("Model loaded.\n")

sentence = "today i moved the drone 12 to storage room 3 and cleaned some dust on top of the box 147, now its clean"
words = sentence.split()

# Function/connecting words — kept as-is, not transformed
CONNECTING_WORDS = {
    "i", "a", "an", "the", "to", "and", "or", "but", "of", "in", "on",
    "at", "by", "for", "with", "from", "up", "into", "now", "its", "it",
    "some", "is", "was", "are", "be", "been", "has", "have", "had",
    "this", "that", "these", "those", "my", "your", "his", "her", "our",
    "not", "no", "so", "if", "as", "do", "did", "will", "would", "could",
    "should", "may", "might", "can", "then", "than", "there",
}

# Normalize consciousness vector to unit length
consciousness_vec = model["consciousness"]
consciousness_unit = consciousness_vec / np.linalg.norm(consciousness_vec)

print(f"Original : {sentence}\n")

for alpha in [0.01, 0.1, 0.2, 0.4, 0.8]:
    result = []
    for word in words:
        stripped = word.strip(".,!")
        suffix = word[len(stripped):]

        # Keep connecting words, numbers, and OOV tokens unchanged
        if stripped.lower() in CONNECTING_WORDS or stripped.replace(".", "").isdigit() or stripped not in model:
            result.append(word)
            continue

        word_vec = model[stripped]
        word_unit = word_vec / np.linalg.norm(word_vec)
        # Shift in the -consciousness direction by alpha
        v_mod = word_unit - alpha * consciousness_unit
        # Nearest neighbor excluding the word itself and connecting words
        similar = model.similar_by_vector(v_mod, topn=50)
        nearest = next(
            (w for w, _ in similar if w != stripped and w.lower() not in CONNECTING_WORDS),
            stripped,
        )
        result.append(nearest + suffix)

    print(f"alpha={alpha:<4}: {' '.join(result)}")

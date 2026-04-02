import gensim.downloader as api
import numpy as np

print("Loading GloVe model (glove-wiki-gigaword-100)...")
model = api.load("glove-wiki-gigaword-100")
print("Model loaded.\n")

sentence = "today i moved the drone 12 to storage room 3 and cleaned some dust on top of the box 147, now its clean"
words = sentence.split()

# Normalize consciousness vector to unit length so alpha is a consistent perturbation scale
consciousness_vec = model["consciousness"]
consciousness_unit = consciousness_vec / np.linalg.norm(consciousness_vec)

print(f"Original : {sentence}\n")

for alpha in [0.01, 0.1, 0.2, 0.4, 0.8]:
    result = []
    for word in words:
        stripped = word.strip(".,!")
        suffix = word[len(stripped):]
        if stripped in model:
            word_vec = model[stripped]
            word_unit = word_vec / np.linalg.norm(word_vec)
            # Subtract alpha * consciousness direction from the unit word vector
            v_mod = word_unit - alpha * consciousness_unit
            # Find nearest neighbor excluding the original word itself
            similar = model.similar_by_vector(v_mod, topn=20)
            nearest = next((w for w, _ in similar if w != stripped), stripped)
            result.append(nearest + suffix)
        else:
            result.append(word)  # keep OOV tokens (numbers, etc.) as-is
    print(f"alpha={alpha:<4}: {' '.join(result)}")

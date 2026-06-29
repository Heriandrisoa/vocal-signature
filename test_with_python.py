from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F
import math

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vector size mismatch")

    dot = 0.0
    na = 0.0
    nb = 0.0

    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]

    denom = math.sqrt(na) * math.sqrt(nb)

    if denom == 0:
        return 0.0

    return dot / denom

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)
emb1 = classifier.encode_batch(
    classifier.load_audio("data/bonjour.wav")
)[0, 0].tolist()

emb2 = classifier.encode_batch(
    classifier.load_audio("data/salama.wav")
)[0, 0].tolist()
print(cosine_similarity(emb1, emb2))
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

print(classifier.mods)
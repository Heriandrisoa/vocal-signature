import torch
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

class SpeechBrainEmbedding(torch.nn.Module):

    def __init__(self, classifier):
        super().__init__()

        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model

    def forward(self, wav):

        lengths = torch.ones(
            wav.shape[0],
            device=wav.device
        )

        feats = self.compute_features(wav)

        feats = self.mean_var_norm(
            feats,
            lengths
        )

        emb = self.embedding_model(
            feats
        )

        return emb
    
model = SpeechBrainEmbedding(classifier)
model.eval()

wav = torch.randn(1, 48000)

emb = model(wav)

print(emb.shape)

torch.onnx.export(
    model,
    wav,
    "revised_model.onnx",
    input_names=["waveform"],
    output_names=["embedding"],
    dynamic_axes={
        "waveform": {
            1: "num_samples"
        }
    },
    opset_version=17
)
import torch
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(    
    source="speechbrain/spkrec-ecapa-voxceleb"
)

model = classifier.mods.embedding_model
model.eval()

dummy_input = torch.randn(1, 300, 80)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["features"],
    output_names=["embedding"],
    opset_version=11
)
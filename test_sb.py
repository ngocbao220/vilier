import numpy as np
import torch
from speechbrain.inference.separation import SepformerSeparation

model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="pretrained_models/sepformer-wsj02mix", run_opts={"device": "cpu"})
audio_8k = np.random.randn(16000).astype(np.float32)
mixture = torch.tensor(audio_8k).unsqueeze(0)
print(mixture.shape)
out = model.separate_batch(mixture)
print(out.shape)

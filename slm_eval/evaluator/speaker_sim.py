# Copyright 2025 Xiaomi Corporation.
from stopes.eval.vocal_style_similarity.vocal_style_sim_tool import get_embedder, compute_cosine_similarity
from torchaudio.transforms import Resample
import torch
import numpy as np

SAMPLE_RATE = 16000


class SpeakerSimEvaluator(torch.nn.Module):

    def __init__(self, model_path, sampling_rate):
        super().__init__()
        self.name = 'speaker_sim'
        self.embedder = get_embedder(model_name="valle", model_path=model_path)
        if sampling_rate != SAMPLE_RATE:
            self.resample = Resample(orig_freq=sampling_rate, new_freq=SAMPLE_RATE)
        else:
            self.resample = None
        self.register_buffer('x', torch.tensor([1]))

    @torch.no_grad()
    def forward(self, xs, x_hats, device):
        if not self.resample is None:
            for i in range(len(xs)):
                xs[i] = self.resample(xs[i].to(self.x.device))
                x_hats[i] = self.resample(x_hats[i].to(self.x.device))
        x_embs = self.embedder(xs, device=device)
        x_hat_embs = self.embedder(x_hats, device=device)
        sims = compute_cosine_similarity(x_embs, x_hat_embs)
        return np.sum(sims), len(sims)
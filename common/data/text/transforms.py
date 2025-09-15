import torch
import torchaudio
from torch import nn
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor, AutoModel
from sentence_transformers import SentenceTransformer


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        """
        Taken from torch documentation:
        https://docs.pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html#generating-transcripts

        :param labels:
        :param blank:
        """
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


class Wav2VecExtractFromAudio(nn.Module):
    def __init__(self, fs: int, device=None):
        super(Wav2VecExtractFromAudio, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().to(self.device)
        self.decoder = GreedyCTCDecoder(labels=bundle.get_labels())

        if fs != bundle.sample_rate:
            raise ValueError("Be sure tor resample the input to the correct sample rate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batched_input: bool = len(x.shape) > 1
        with torch.inference_mode():
            y, _ = self.model(x.to(self.device))

        transcript = self.decoder(y) if not batched_input else [self.decoder(b) for b in y.unbind(0)]

        return transcript


class Speech2TextExtract(nn.Module):
    def __init__(self, fs: int, model_name="facebook/s2t-medium-mustc-multilingual-st", device=None):
        super(Speech2TextExtract, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        # Needed to extract
        self.model = Speech2TextForConditionalGeneration.from_pretrained(model_name, device_map=self.device)
        self.processor = Speech2TextProcessor.from_pretrained(model_name, device_map=self.device)
        self.fs = fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i = self.processor(x, sampling_rate=self.fs, return_tensors="pt")
        generated_ids = self.model.generate(**i.to(self.device))
        translation = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return translation


class MiniLMEmbedderTransform(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device=None):
        super(MiniLMEmbedderTransform, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        # self.preprocessor
        self.model = SentenceTransformer(model_name, device=self.device)

    def forward(self, x: list[str]) -> torch.Tensor:
        embeddings = self.model.encode(x)
        embeddings = torch.Tensor(embeddings).to(self.device)

        return embeddings

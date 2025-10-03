import json
import logging
import re
from typing import Iterable

import torch
import torchaudio

from sentence_transformers import SentenceTransformer
from torch import nn
from torchaudio.transforms import Resample
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor, AutoProcessor, pipeline, \
    AutoModelForSpeechSeq2Seq, BertModel, BertTokenizer

from core_data.media.audio.transforms import ToMono
from core_data.media.text import Text
from core_data.utils import timed


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

    @timed()
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

    @timed()
    def forward(self, x: torch.Tensor) -> list[str]:
        batched_input: bool = len(x.shape) > 1
        if not batched_input:
            x = x.unsqueeze(0)

        with torch.inference_mode():
            y, _ = self.model(x.to(self.device))

        transcript = [self.decoder(b).replace("|", " ") for b in y.unbind(0)]
        transcript = [re.sub(r'[^A-Za-z0-9 ]+', '', b) for b in transcript]
        return transcript


# We can try openai/whisper-large-v3
class WhisperTextExtractFromAudio(nn.Module):
    def __init__(self, fs: int, device=None, model_id="openai/whisper-large-v3"):
        super(WhisperTextExtractFromAudio, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.torch_dtype = torch.float16 if device != "cpu" else torch.float32

        self.fs = fs
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition", model="openai/whisper-large-v3", return_timestamps=True,
            device=self.device, torch_dtype=self.torch_dtype,
        )

    # TODO: funziona ma senza cuda va stra lento. vedi se caricabile su cuda
    @timed()
    def forward(self, x: torch.Tensor) -> list[str]:
        # 16 kHz mono float32 in [-1, 1]
        if len(x.shape) != 1:
            out = self.pipe(x.numpy(), generate_kwargs={"language": "english"}, batch_size=x.shape[0])
        else:
            out = self.pipe(x.numpy(), generate_kwargs={"language": "english"})
        return out["text"]


class Speech2TextExtract(nn.Module):
    def __init__(self, fs: int, model_name="facebook/s2t-medium-mustc-multilingual-st", device=None):
        super(Speech2TextExtract, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        # Needed to extract
        self.model = Speech2TextForConditionalGeneration.from_pretrained(model_name, device_map=self.device)
        self.processor = Speech2TextProcessor.from_pretrained(model_name, device_map=self.device)
        self.fs = fs

    @timed()
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

    @timed()
    def forward(self, x: list[str]) -> torch.Tensor:
        embeddings = self.model.encode(x)
        embeddings = torch.Tensor(embeddings).to(self.device)

        return embeddings


class TextRegistry(nn.Module):
    def __init__(self, store_path: str):
        """
        Might not be the smartest or best approach, but it just will serve our purpose.
        Not used as we moved store outputs directly as a step coming before main pre-processing.
        Reason for it is that the dataset is used by multiple models so the times and other stuff have to be pre-computed
        in order to make the model samples and teacher (VATE in our case) aligned.
        :param store_path: Where to store the extracted text
        """
        super(TextRegistry, self).__init__()
        self.store_path = store_path

    def forward(self, transcript: str) -> str:
        with open(self.store_path, "a", encoding="utf-8") as f:
            if isinstance(transcript, list):
                f.write(f"{"[MEDIA-DIVIDER]".join(transcript)}\n")
            else:
                f.write(transcript + "\n")

        return transcript


class WhisperExtractor(nn.Module):
    def __init__(self, model_id: str = "openai/whisper-medium", device=None):
        super(WhisperExtractor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.torch_dtype = torch.float16 if device != "cpu" else torch.float16  # torch.float32
        # Parameter that comes from whisper requirements

        self.model_fs: int = 16000
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(  # 1 minuto
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps="word",
            device=self.device,
            torch_dtype=self.torch_dtype
        )

    @timed()
    def forward(self, x: torch.Tensor, fs: int) -> dict:
        aud = ToMono()(x)
        aud = aud.float()
        aud = Resample(orig_freq=fs, new_freq=self.model_fs)(aud)
        aud = aud.numpy()

        try:
            with torch.inference_mode():
                return self.pipe(aud)

        except Exception as e:
            raise e


class WhisperClipTextExtract(nn.Module):
    def __init__(self, model_id: str = "openai/whisper-large-v3", device=None):
        super(WhisperClipTextExtract, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.torch_dtype = torch.float16 if device != "cpu" else torch.float16  # torch.float32
        # Parameter that comes from whisper requirements

        self.model_fs: int = 16000
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(  # 1 minuto
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps="word",
            device=self.device,
            torch_dtype=self.torch_dtype
        )

    @timed()
    def forward(self, txt: Text) -> Text:
        # Apply only once.
        if txt.text_context is not None:
            return txt

        aud = torch.tensor(txt.base_audio.to_soundarray()).float()
        aud = ToMono()(aud)
        aud = Resample(orig_freq=txt.base_audio.fps, new_freq=self.model_fs)(aud)
        try:
            with torch.inference_mode():
                txt.text_context = self.pipe(aud.numpy())
        except Exception as e:
            logging.exception(e)
            raise e

        return txt


class RestoreTextExtract(nn.Module):
    def __init__(self, base_path: str):
        super(RestoreTextExtract, self).__init__()
        self.base_path: str = base_path

    def forward(self, txt: Text) -> Text:
        try:
            with open(self.base_path + txt.eid + "-txt-extract.json") as f:
                txt.text_context = json.load(f)
        except Exception as e:
            logging.error(e)
            logging.warning(
                f"Tried to read text context for {txt.eid} but we had an error.\n "
                f"Empty chunks will be set instead"
            )
            txt.text_context = {"chunks": []}
        return txt


class SubclipTextExtract(nn.Module):

    def __init__(self, interleaved: bool = False, i_max_length: int = None):
        super(SubclipTextExtract, self).__init__()
        if interleaved and i_max_length is None:
            raise ValueError("If using interleaved i_max_length has to be specified")
        self.interleaved: bool = interleaved
        self.i_max_length: int = i_max_length

    @staticmethod
    def chunk_extract(chunk, start, stop):
        ch_start, ch_stop = chunk["timestamp"]
        start_valid = ch_start >= start if ch_start is not None else start >= ch_stop >= stop
        stop_valid = ch_stop <= stop if ch_stop is not None else start <= ch_start <= stop
        return chunk["text"] if start_valid and stop_valid else ""

    # noinspection PyMethodMayBeStatic
    def forward(self, x: Text) -> list[str]:
        start, stop = x.interval

        if self.interleaved:
            i_segments = []
            length = stop - start
            segments = int(length / self.i_max_length) + (length % self.i_max_length != 0)

            for i in range(segments):
                txt = ""
                i_start, i_stop = i * self.i_max_length + start, (i + 1) * self.i_max_length + start
                if i_stop > stop: i_stop = stop
                for chunk in x.text_context["chunks"]:
                    txt += self.chunk_extract(chunk, i_start, i_stop)
                i_segments.append(txt)

            return i_segments

        txt = ""
        for chunk in x.text_context["chunks"]:
            txt += self.chunk_extract(chunk, start, stop)
        return [txt]


class BertEmbeddings(nn.Module):
    def __init__(self, device=None):
        super(BertEmbeddings, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", device_map=self.device)

    def forward(self, x: list[str]):
        item_txt = torch.tensor([self.tokenizer.encode(x)])
        item_txt = item_txt.to(self.device)
        with torch.inference_mode():
            y = self.model(item_txt).pooler_output.squeeze(0)
        return y

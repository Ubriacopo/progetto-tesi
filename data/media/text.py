import torch
from torch import Tensor
from transformers import BertTokenizer, BertModel

from data.media.media import Media


class Text(Media):
    # We use BERT to handle text data. These are fixed by VATE
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Yes, it's a sentence embedding—but not the best for all tasks.
    # Use sentence-transformers for more meaningful embeddings.
    # model = SentenceTransformer('all-mpnet-base-v2') -> Requires to retrain VATE tho
    text_model = BertModel.from_pretrained("bert-base-uncased")

    def __init__(self, file_path: str):
        super().__init__(file_path)
        # We are lazy loaders.
        self.text: str | None = None
        self.processed_text: Tensor | None = None  # TODO No idea what type this is

    def get_info(self):
        return {"file_path": self.file_path}

    def _inner_load(self, **kwargs):
        with open(self.file_path, "r") as f:
            self.text = f.read()

    def _inner_process(self, **kwargs):
        embeddings = torch.tensor([Text.text_tokenizer.encode(self.text)])
        with torch.no_grad():
            # TODO: Should try and understand what this does and why.
            self.processed_text = Text.text_model(embeddings).pooler_output.squeeze(0)

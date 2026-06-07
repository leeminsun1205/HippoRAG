from typing import List
import json

import torch
import numpy as np

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from sentence_transformers import SentenceTransformer

class TransformersEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "Transformers/"
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name[len("Transformers/"):]
        self.embedding_type = 'float'
        self.batch_size = 64

        self.model = SentenceTransformer(self.model_id, device = "cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, texts: List[str], norm: bool = True) -> np.ndarray:
        try:
            # SentenceTransformer batches internally via `batch_size`.
            # `normalize_embeddings` is False by default, which is wrong for
            # cosine-style retrieval (HippoRAG scores with dot product), so we
            # pass it explicitly.
            response = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=norm,
                show_progress_bar=False,
            )
        except Exception as err:
            raise Exception(f"An error occurred: {err}")
        return np.array(response)

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # Instruction is applied to the *query* side only (HippoRAG passes it
        # when encoding queries; passage/entity/fact insertion passes none),
        # preserving the asymmetric query-vs-passage encoding these models
        # expect. Mirrors NVEmbedV2's handling.
        instruction = kwargs.get("instruction", "")
        if instruction:
            texts = [f"{instruction}\n{text}" for text in texts]

        # Respect requested normalization, defaulting to the global config.
        # Required so dot-product scoring behaves as cosine similarity.
        norm = kwargs.get("norm", self.global_config.embedding_return_as_normalized)

        return self.encode(texts, norm=norm)

import os
import requests
import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


class JinaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="jinaai/jina-embeddings-v2-base-en") -> None:
        self.model = SentenceTransformer(model_name, trust_remote_code=True) # trust_remote_code is needed to use the encode method

    def create_embedding(self, text):
        return self.model.encode(text)
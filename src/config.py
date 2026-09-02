"""Config loader with Pydantic validation."""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class AppConfig(BaseModel):
    name: str = "Digital Islamic Library AI"
    log_level: str = "INFO"


class EmbeddingConfig(BaseModel):
    model_name: str = "gemini-embedding-2-preview"
    output_dimensionality: int = 768
    api_key_env: str = "GEMINI_API_KEY"


class LLMConfig(BaseModel):
    model_name: str = "gemini-3.6-flash"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.1
    max_output_tokens: int = 2048


class GeminiConfig(BaseModel):
    api_key_env: str = "GEMINI_API_KEY"
    llm_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-2-preview"


class GroqConfig(BaseModel):
    api_key_env: str = "GROQ_API_KEY"
    llm_model: str = "llama-3.3-70b-versatile"


class NvidiaConfig(BaseModel):
    api_key_env: str = "NVIDIA_API_KEY"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = "deepseek-ai/deepseek-v4-flash-0731"
    embedding_model: str = "nemotron-3-embed-1b"


class CrossEncoderConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ModelsConfig(BaseModel):
    provider: str = "groq"  # "groq" | "gemini" | "nvidia"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    nvidia: NvidiaConfig = Field(default_factory=NvidiaConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)


class PineconeConfig(BaseModel):
    api_key_env: str = "PINECONE_API_KEY"
    index_name: str = "advanced-rag-index"
    namespace: str = "default"
    cloud: str = "aws"
    region: str = "us-east-1"


class ChunkingConfig(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 60
    separators: List[str] = Field(default_factory=lambda: [
        "\n## ", "\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", " "
    ])


class RetrievalConfig(BaseModel):
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    fusion_top_k: int = 50
    rerank_top_k: int = 5


class PathsConfig(BaseModel):
    bm25_index: str = "./data/bm25_index.pkl"
    documents_dir: str = "./data/pdfs"


class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_gemini_api_key(self) -> str:
        return os.environ.get(self.models.gemini.api_key_env, os.environ.get("GEMINI_API_KEY", ""))

    def get_groq_api_key(self) -> str:
        return os.environ.get(self.models.groq.api_key_env, os.environ.get("GROQ_API_KEY", ""))

    def get_nvidia_api_key(self) -> str:
        return os.environ.get(self.models.nvidia.api_key_env, os.environ.get("NVIDIA_API_KEY", ""))

    def get_nvidia_base_url(self) -> str:
        return os.environ.get("NVIDIA_BASE_URL", self.models.nvidia.base_url)

    def get_pinecone_api_key(self) -> str:
        return os.environ.get(self.pinecone.api_key_env, os.environ.get("PINECONE_API_KEY", ""))


CONFIG = Config.from_yaml()

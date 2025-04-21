# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from typing import Union
from pydantic import BaseModel


class OpenAiLlmConfigData(BaseModel):
    """Base class for OpenAI Llm configuration data"""
    open_ai_url: str = None
    open_ai_key: str = None
    open_ai_version: str = None
    prompt_template: str = None
    completion_model: str = None
    temperature: float = None


class OpenSourceLlmConfigData(BaseModel):
    """Base class for OpenSource Llm configuration data"""
    model_name: str = None
    model_path: str = None
    cache_dir: str = None


class OpenAiEmbeddingConfigData(BaseModel):
    """Base class for OpenAI Embedding configuration data"""
    open_ai_url: str = None
    open_ai_key: str = None
    open_ai_version: str = None
    model_name: str = None


class OpenSourceEmbeddingConfigData(BaseModel):
    """Base class for OpenSource Embedding configuration data"""
    model_name: str = None
    model_path: str = None


class AudioToTextConfigData(BaseModel):
    """Base class for request to Audio to Text """
    audio_converter_model_path: str = None
    compressed_audio_filepath: str = None
    audio_chunk_length: int = None
    audio_target_sample_rate: int = None
    audio_batch_size: int = None
    open_ai_config: OpenAiLlmConfigData = None


class CodeTranslatorConfigData(BaseModel):
    """Base class for request to code translator"""
    open_ai_config: OpenAiLlmConfigData = None
    token_limit: int = None
    vocab_dir_path: str = None


class SummarizerConfigData(BaseModel):
    """Base class for request to summarize"""
    open_ai_config: OpenAiLlmConfigData = None
    summary_tokens: int = None
    word_limit: int = None
    token_group_size: int = None
    vocab_dir_path: str = None
    chunk_length: int = None


class AddSearchableEmbeddingsConfigData(BaseModel):
    """Base class for request to add searchable embeddings"""
    embedding_config: Union[OpenAiEmbeddingConfigData,
                            OpenSourceEmbeddingConfigData] = None
    chunk_length: int = None
    token_group_size: int = None
    vocab_dir_path: str = None
    vector_db_directory: str = None
    chroma_db_impl: str = None


class ContextRetrieverConfigData(BaseModel):
    """Base class for request to retrieve context"""
    embedding_config: Union[OpenAiEmbeddingConfigData,
                            OpenSourceEmbeddingConfigData] = None
    vector_db_directory: str = None
    context_length: int = None
    top_k_matches: int = None
    threshold_distance: float = None


class AnswerGeneratorConfigData(BaseModel):
    """Base class for request to generate answer"""
    embedding_config: Union[OpenAiEmbeddingConfigData,
                            OpenSourceEmbeddingConfigData] = None
    llm_config: Union[OpenAiLlmConfigData, OpenSourceLlmConfigData] = None
    vector_db_directory: str = None
    context_length: int = None
    top_k_matches: int = None
    threshold_distance: float = None


class DocumentResetterConfigData(BaseModel):
    """Base class for request to reset document"""
    vector_db_directory: str = None
    collection_name: str = None


class VideoMomentRetrieverConfigData(BaseModel):
    """Base class for request to retrieve video moments"""
    output_root_path: str = None
    opensource_llm_config: OpenSourceLlmConfigData = None
    similarity_threshold: float = None


class KnowledgeGraphBuilderConfigData(BaseModel):
    """Base class for request to build knowledge graph"""
    root_path: str = None
    output_file_prefix: str = None
    openai_llm_config: OpenAiLlmConfigData = None


class SearchKnowledgeGraphBuilderConfigData(BaseModel):
    """Base class for request to build knowledge graph"""
    root_path: str = None
    output_file_prefix: str = None
    openai_llm_config: OpenAiLlmConfigData = None

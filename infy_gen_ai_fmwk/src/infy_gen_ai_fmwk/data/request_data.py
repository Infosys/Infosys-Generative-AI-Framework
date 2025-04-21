# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from enum import Enum
from pydantic import BaseModel
from typing import Optional


class AudioToTextRequestData(BaseModel):
    """Base class for request to Audio to Text """
    file_path: str = None
    is_mom_required: bool = True


class CodeTranslatorRequestData(BaseModel):
    """Base class for request to code translator"""
    src_code_file_path: str = None
    src_lang: str = None
    tgt_lang: str = None


class CodeDocumentRequestData(BaseModel):
    """Base class for request to code document"""
    src_code_file_path: str = None
    lang: str = None


class UnitTestCreatorRequestData(BaseModel):
    """Base class for request to create unit test"""
    src_code_file_path: str = None
    lang: str = None


class MultipleDocsGeneratorRequestData(BaseModel):
    """Base class for request to generate multiple docs"""
    src_code_file_paths: list = None
    output_compressed_file_path: str = None


class SummarizerRequestData(BaseModel):
    """Base class for request to summarize"""
    file_path: str = None
    summarization_type: str = None


class InsightsGeneratorRequestData(BaseModel):
    """Base class for request to generate insights"""
    file_path: str = None
    completions: int = None


class MetaAndDescriptionGeneratorRequestData(BaseModel):
    """Base class for request to generate meta and description"""
    file_path: str = None


class ModelType(Enum):
    OPENAI = 'openai'
    OPENSOURCE = 'opensource'


class AddSearchableEmbeddingsRequestData(BaseModel):
    """Base class for request to add searchable embeddings """
    file_paths: list = None
    embedding_type: ModelType = None


class AnswerGeneratorRequestData(BaseModel):
    """Base class for request to generate answer"""
    query: str = None
    embedding_type: ModelType = None
    rag: bool = True


class ContextRetrieverRequestData(BaseModel):
    """Base class for request to retrieve context"""
    query: str = None
    embedding_type: ModelType = None


class VideoMomentRetrieverRequestData(BaseModel):
    """Base class for request to retrieve video moment"""
    query: str = None
    display_results_count: int = None
    video_file_path: str = None
    clip_duration: int = None


class ModeType(Enum):
    FILTER = 'FILTER'
    UPDATE = 'UPDATE'
    filter = 'filter'
    update = 'update'


class KnowledgeGraphBuilderRequestData(BaseModel):
    """Base class for request to build knowledge graph"""
    query: str = None
    mode: ModeType = None
    graph_json_file_path: Optional[str] = None


class SearchKnowledgeGraphBuilderRequestData(BaseModel):
    """Base class for request to search knowledge graph"""
    query: str = None
    graph_json_file_path: Optional[str] = None

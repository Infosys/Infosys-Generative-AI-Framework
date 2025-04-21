# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from pydantic import BaseModel


class AudioToTextResponseData(BaseModel):
    """Base class for response from Audio to Text """
    transcription_response_txt: str = None
    mom_text: str = ""


class CodeTranslatorResponseData(BaseModel):
    """Base class for response from code translator"""
    translated_code: str = None


class CodeDocumentResponseData(BaseModel):
    """Base class for response from code translator"""
    code_document_response: str = None


class UnitTestCreatorResponseData(BaseModel):
    """Base class for response from create unit test"""
    unit_test_response: str = None


class MultipleDocsGeneratorResponseData(BaseModel):
    """Base class for response from generate multiple docs"""
    code_docs_response: str = None
    rejected_files: list = None


class SummarizerResponseData(BaseModel):
    """Base class for response from summarize"""
    summary: str = None


class InsightsGeneratorResponseData(BaseModel):
    """Base class for response from generate insights"""
    insights_response: str = None


class MetaAndDescriptionGeneratorResponseData(BaseModel):
    """Base class for response from generate meta and description"""
    meta_descr_response: str = None


class AddSearchableEmbeddingsResponseData(BaseModel):
    """Base class for response from add searchable embeddings """
    content: str = None
    rejected_files: list = None


class AnswerGeneratorResponseData(BaseModel):
    """Base class for response from generate answer"""
    generated_answer: str = None


class ContextRetrieverResponseData(BaseModel):
    """Base class for response from retrieve context"""
    closest_matches: list = None


class DocumentResetterResponseData(BaseModel):
    """Base class for response from reset document"""
    content: str = None


class VideoMomentRetrieverResponseData(BaseModel):
    """Base class for response from retrieve video moments"""
    output_file_paths: list = None


class KnowledgeGraphBuilderResponseData(BaseModel):
    """Base class for response from build knowledge graph"""
    graph_output_image_file_path: str = None
    graph_output_json_file_path: str = None
    message_text: str = None


class SearchKnowledgeGraphBuilderResponseData(BaseModel):
    """Base class for response from search knowledge graph"""
    graph_output_image_file_path: str = None
    message_text: str = None

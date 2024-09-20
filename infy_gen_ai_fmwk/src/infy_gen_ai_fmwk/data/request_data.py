# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from pydantic import BaseModel

class AudioToTextRequestData(BaseModel):
    """Base class for request to Audio to Text """
    file_path: str = None
    is_mom_required: bool = True


class AudioToTextResponseData(BaseModel):
    """Base class for response from Audio to Text """
    transcription_response_txt: str = None
    mom_text: str = None


class CodeTranslatorRequestData(BaseModel):
    """Base class for request to code translator"""
    src_code_file_path: str = None
    src_lang: str = None
    tgt_lang: str = None


class CodeTranslatorResponseData(BaseModel):
    """Base class for response from code translator"""
    translated_code: str = None


class CodeDocumentRequestData(BaseModel):
    """Base class for request to code document"""
    src_code_file_path: str = None
    lang: str = None


class CodeDocumentResponseData(BaseModel):
    """Base class for response from code translator"""
    code_document_response: str = None


class UnitTestCreatorRequestData(BaseModel):
    """Base class for request to create unit test"""
    src_code_file_path: str = None
    lang: str = None


class UnitTestCreatorResponseData(BaseModel):
    """Base class for response from create unit test"""
    unit_test_response: str = None


class MultipleDocsGeneratorRequestData(BaseModel):
    """Base class for request to generate multiple docs"""
    src_code_file_paths: list = None
    output_compressed_file_path: str = None


class MultipleDocsGeneratorResponseData(BaseModel):
    """Base class for response from generate multiple docs"""
    code_docs_response: str = None
    rejected_files: list = None


class SummarizerRequestData(BaseModel):
    """Base class for request to summarize"""
    file_path: str = None
    summarization_type: str = None


class SummarizerResponseData(BaseModel):
    """Base class for response from summarize"""
    summary: str = None


class InsightsGeneratorRequestData(BaseModel):
    """Base class for request to generate insights"""
    file_path: str = None
    completions: int = None


class InsightsGeneratorResponseData(BaseModel):
    """Base class for response from generate insights"""
    insights_response: str = None


class MetaAndDescriptionGeneratorRequestData(BaseModel):
    """Base class for request to generate meta and description"""
    file_path: str = None


class MetaAndDescriptionGeneratorResponseData(BaseModel):
    """Base class for response from generate meta and description"""
    meta_descr_response: str = None


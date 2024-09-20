# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from pydantic import BaseModel

class OpenAiLlmConfigData(BaseModel):
    """Base class for OpenAI Llm configuration data"""
    open_ai_url: str = None
    open_ai_key: str = None
    open_ai_version: str = None
    prompt_template: str = None
    completion_model: str = None
    temperature: float = None

class AudioToTextConfigData(BaseModel):
    """Base class for request to Audio to Text """
    audio_converter_model_path: str = None
    compressed_audio_filepath: str = None
    audio_chunck_length: int = None
    audio_Target_sr: int = None
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



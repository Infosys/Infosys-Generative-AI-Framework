# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData, SummarizerConfigData
from infy_gen_ai_fmwk.data.request_data import SummarizerRequestData
from infy_gen_ai_fmwk.service.summarize_service import SummarizeService


def test_summarize_valid_extension_with_recursive_type():
    # Input file path should be updated by the user
    test_file_path = ""
    config_data = SummarizerConfigData(
        summary_tokens=300,
        word_limit=1500,
        token_group_size=2048,
        chunk_length=800,
        #Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
        vocab_dir_path="C:/MyProgramFiles/AI/models/gpt2"
    )

    request_data = SummarizerRequestData()
    request_data.file_path = test_file_path
    request_data.summarization_type = "recursive"
    config_data.open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Turn this text into a short summary:\n---\n\n{text} \n\n---\n\nSummary:"
    )

    summary_service = SummarizeService(config_data)
    result = summary_service.summarize(request_data)
    assert result is not None and result.summary is not None


def test_summarize_valid_extension_with_linear_type():
    # Input file path should be updated by the user
    test_file_path = ""
    config_data = SummarizerConfigData(
        summary_tokens=300,
        word_limit=1500,
        token_group_size=2048,
        chunk_length=800,
        #Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
        vocab_dir_path="C:/MyProgramFiles/AI/models/gpt2"
    )

    request_data = SummarizerRequestData()
    request_data.file_path = test_file_path
    request_data.summarization_type = "linear"
    config_data.open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Turn this text into a short summary:\n---\n\n{text} \n\n---\n\nSummary:"
    )

    summary_service = SummarizeService(config_data)
    result = summary_service.summarize(request_data)
    assert result is not None and result.summary is not None


def test_summarize_invalid_extension():
    # Input file path having invalid extension can be updated by the user
    test_file_path = "data/sample/input/MS-Dhoni-Biography.docx 2.mp3"
    config_data = SummarizerConfigData(
        summary_tokens=300,
        word_limit=1500,
        token_group_size=2048,
        chunk_length=800,
        #Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
        vocab_dir_path="C:/MyProgramFiles/AI/models/gpt2"
    )

    request_data = SummarizerRequestData()
    request_data.file_path = test_file_path
    request_data.summarization_type = "recursive"
    config_data.open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Turn this text into a short summary:\n---\n\n{text} \n\n---\n\nSummary:"
    )

    summary_service = SummarizeService(config_data)
    with pytest.raises(ValueError):
        summary_service.summarize(request_data)

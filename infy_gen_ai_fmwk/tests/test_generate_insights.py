# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.service.generate_insights_service import GenerateInsights
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import InsightsGeneratorRequestData

def test_generate_insights_with_xlsx_file():
    # Input file path should be updated by the user
    test_file_path = ""
    request_data = InsightsGeneratorRequestData()
    request_data.file_path = test_file_path
    request_data.completions = 5
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template='Generate detailed meaningful insights from the given {data} in 5-10 sentences\n'
    )

    summary_service = GenerateInsights(config_data)
    result = summary_service.generate_insights(request_data)
    assert result is not None and result.insights_response is not None


def test_generate_insights_with_csv_file():
    # Input file path should be updated by the user
    test_file_path = ""
    request_data = InsightsGeneratorRequestData()
    request_data.file_path = test_file_path
    request_data.completions = 5
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template='Generate detailed meaningful insights from the given {data} in 5-10 sentences\n'
    )

    summary_service = GenerateInsights(config_data)
    result = summary_service.generate_insights(request_data)
    assert result is not None and result.insights_response is not None


def test_generate_insights_with_invalid_extension():
    # Input file path having invalid extension can be updated by the user
    test_file_path = "data/sample/input/test1_java.txt"
    request_data = InsightsGeneratorRequestData()
    request_data.file_path = test_file_path
    request_data.completions = 5
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template='Generate detailed meaningful insights from the given {data} in 5-10 sentences\n'
    )

    summary_service = GenerateInsights(config_data)
    with pytest.raises(ValueError):
        summary_service.generate_insights(request_data)

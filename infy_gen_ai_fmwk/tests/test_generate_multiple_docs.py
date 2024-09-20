# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import MultipleDocsGeneratorRequestData
from infy_gen_ai_fmwk.service.generate_multiple_docs_service import GenerateMultipleDocsService

def test_generate_multi_docs_with_one_invalid_ext():
    # Input file path should be updated by the user
    test_file_path = ""
    request_data = MultipleDocsGeneratorRequestData()
    request_data.src_code_file_paths = test_file_path
    # output compressed file path to be updated by the user
    request_data.output_compressed_file_path = "c:/test/output"
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate business documentation for the following code\nCode:\n{contents}\n\nDocs"
    )

    code_service = GenerateMultipleDocsService(config_data)
    result = code_service.generate_multi_docs(request_data)
    assert result is not None and result.code_docs_response is not None

def test_generate_multi_docs_with_all_valid_ext():
    # Input file path should be updated by the user
    test_file_path = ""
    request_data = MultipleDocsGeneratorRequestData()
    request_data.src_code_file_paths = test_file_path
    # output compressed file path to be updated by the user
    request_data.output_compressed_file_path = "c:/test/output"
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate business documentation for the following code\nCode:\n{contents}\n\nDocs"
    )

    code_service = GenerateMultipleDocsService(config_data)
    result = code_service.generate_multi_docs(request_data)
    assert result is not None and result.code_docs_response is not None

def test_generate_multi_docs_invalid_ext():
    # Input file path having invalid extension can be updated by the user
    test_file_path = "data/sample/input/test1_java.mp3"
    request_data = MultipleDocsGeneratorRequestData()
    request_data.src_code_file_paths = test_file_path
    # output compressed file path to be updated by the user
    request_data.output_compressed_file_path = "c:/test/output"
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate business documentation for the following code\nCode:\n{contents}\n\nDocs"
    )

    code_service = GenerateMultipleDocsService(config_data)
    with pytest.raises(ValueError):
        code_service.generate_multi_docs(request_data)    

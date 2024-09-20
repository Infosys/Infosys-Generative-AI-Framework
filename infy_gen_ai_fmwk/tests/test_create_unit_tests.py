# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import UnitTestCreatorRequestData
from infy_gen_ai_fmwk.service.create_unit_tests_service import CreateUnitTestsService

def test_create_unit_tests_lang_java_txt():
    # Input file path should be updated by the user  
    test_file_path = ""
    request_data = UnitTestCreatorRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.lang = 'java'
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate source code for 6 {lang} unit test cases for the given {lang} code. The test cases should be numbered\n"

    )

    code_service = CreateUnitTestsService(config_data)
    result = code_service.create_unit_tests(request_data)
    assert result is not None and result.unit_test_response is not None

def test_create_unit_tests_lang_python():
    # Input file path should be updated by the user  
    test_file_path = ""
    request_data = UnitTestCreatorRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.lang = 'python'
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate source code for 6 {lang} unit test cases for the given {lang} code. The test cases should be numbered\n"
    )

    code_service = CreateUnitTestsService(config_data)
    result = code_service.create_unit_tests(request_data)
    assert result is not None and result.unit_test_response is not None

def test_create_unit_tests_invalid_ext():
    # Input file path having invalid extension can be updated by the user
    test_file_path = "data/sample/input/test1_java.mp3"
    request_data = UnitTestCreatorRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.lang = 'java'
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate source code for 6 {lang} unit test cases for the given {lang} code. The test cases should be numbered\n"
    )

    code_service = CreateUnitTestsService(config_data)
    with pytest.raises(ValueError):
        code_service.create_unit_tests(request_data)

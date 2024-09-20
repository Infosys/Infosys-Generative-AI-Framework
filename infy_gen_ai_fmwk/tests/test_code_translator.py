# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.data.request_data import CodeTranslatorRequestData
from infy_gen_ai_fmwk.service.code_translator_service import CodeTranslatorService
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData, CodeTranslatorConfigData


def test_translate_code_src_java_text():
    # Input file path should be updated by the user
    test_file_path = ""
    config_data = CodeTranslatorConfigData()
    request_data = CodeTranslatorRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.src_lang = 'java'
    request_data.tgt_lang = 'c'
    config_data.token_limit = 2000
    #Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
    config_data.vocab_dir_path = "C:/MyProgramFiles/AI/models/gpt2"
    config_data.open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Convert the following {src_lang} code to {tgt_lang} code:\n\n"
    )

    code_service = CodeTranslatorService(config_data)
    result = code_service.translate_code(request_data)
    assert result is not None and result.translated_code is not None

def test_translate_code_src_java_lower():
    # Input file path should be updated by the user
    test_file_path = ""
    config_data = CodeTranslatorConfigData()
    request_data = CodeTranslatorRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.src_lang = 'java'
    request_data.tgt_lang = 'python'
    config_data.token_limit = 2000
    #Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
    config_data.vocab_dir_path = "C:/MyProgramFiles/AI/models/gpt2"
    config_data.open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Convert the following {src_lang} code to {tgt_lang} code:\n\n"
    )

    code_service = CodeTranslatorService(config_data)
    result = code_service.translate_code(request_data)
    assert result is not None and result.translated_code is not None


def test_translate_code_src_java_upper():
    # Input file path should be updated by the user
    test_file_path = ""
    config_data = CodeTranslatorConfigData()
    request_data = CodeTranslatorRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.src_lang = 'java'
    request_data.tgt_lang = 'c#'
    config_data.token_limit = 2000
    #Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
    config_data.vocab_dir_path = "C:/MyProgramFiles/AI/models/gpt2"
    config_data.open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Convert the following {src_lang} code to {tgt_lang} code:\n\n"
    )

    code_service = CodeTranslatorService(config_data)
    result = code_service.translate_code(request_data)
    assert result is not None and result.translated_code is not None


def test_translate_code_src_python_lower():
    # Input file path should be updated by the user
    test_file_path = ""
    config_data = CodeTranslatorConfigData()
    request_data = CodeTranslatorRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.src_lang = 'python'
    request_data.tgt_lang = 'c'
    config_data.token_limit = 2000
    #Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
    config_data.vocab_dir_path = "C:/MyProgramFiles/AI/models/gpt2"
    config_data.open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Convert the following {src_lang} code to {tgt_lang} code:\n\n"
    )

    code_service = CodeTranslatorService(config_data)
    result = code_service.translate_code(request_data)
    assert result is not None and result.translated_code is not None

def test_translate_code_invalid_ext():
    # Input file path having invalid extension can be updated by the user
    test_file_path = "data/sample/input/java.mp3"
    config_data = CodeTranslatorConfigData()
    request_data = CodeTranslatorRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.src_lang = 'java'
    request_data.tgt_lang = 'c'
    config_data.token_limit = 2000
    #Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
    config_data.vocab_dir_path = "C:/MyProgramFiles/AI/models/gpt2"
    config_data.open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Convert the following {src_lang} code to {tgt_lang} code:\n\n"
    )

    code_service = CodeTranslatorService(config_data)
    with pytest.raises(ValueError):
        code_service.translate_code(request_data)    

# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import CodeDocumentRequestData
from infy_gen_ai_fmwk.service.code_document_service import CodeDocumentService

def test_code_document_lang_java_txt():
    # Input file path should be updated by the user
    test_file_path = ""
    request_data = CodeDocumentRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.lang = 'java'
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Document and regenerate the following {lang} code with method definitions:\nCode:\n{contents}\n Docs:"
    )
    code_service = CodeDocumentService(config_data)
    result = code_service.generate_code_docs(request_data)
    assert result is not None and result.code_document_response is not None

def test_code_document_lang_python_lower():
    # Input file path should be updated by the user 
    test_file_path = ""
    request_data = CodeDocumentRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.lang = 'python'
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Document and regenerate the following {lang} code with method definitions:\nCode:\n{contents}\n Docs:"
    )

    code_service = CodeDocumentService(config_data)
    result = code_service.generate_code_docs(request_data)
    assert result is not None and result.code_document_response is not None


def test_code_document_lang_python_upper():
    # Input file path should be updated by the user 
    test_file_path = ""
    request_data = CodeDocumentRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.lang = 'python'
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Document and regenerate the following {lang} code with method definitions:\nCode:\n{contents}\n Docs:"
    )

    code_service = CodeDocumentService(config_data)
    result = code_service.generate_code_docs(request_data)
    assert result is not None and result.code_document_response is not None

def test_code_document_invalid_ext():
    # Input file path having invalid extension can be updated by the user
    test_file_path = "data/sample/input/test1_java.mp3"
    request_data = CodeDocumentRequestData()
    request_data.src_code_file_path = test_file_path
    request_data.lang = 'java'
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Document and regenerate the following {lang} code with method definitions:\nCode:\n{contents}\n Docs:"
    )

    code_service = CodeDocumentService(config_data)
    with pytest.raises(ValueError):
        code_service.generate_code_docs(request_data)    

# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.service.answer_generator_service import AnswerGeneratorService
from infy_gen_ai_fmwk.data.config_data import AnswerGeneratorConfigData, OpenAiEmbeddingConfigData, OpenAiLlmConfigData, OpenSourceEmbeddingConfigData, OpenSourceLlmConfigData
from infy_gen_ai_fmwk.data.request_data import AnswerGeneratorRequestData


def test_generate_answer_openai_rag():
    request_data = AnswerGeneratorRequestData(
        # Query should be updated by the user
        query="",
        # Embedding type should be updated by the user
        embedding_type="openai",
        # Rag should be updated by the user
        rag=True
    )
    config_data = AnswerGeneratorConfigData(
        # Vector_db_directory should be updated by the user
        vector_db_directory="C:/test/vectordb",
        context_length=1000,
        # Top_k_matches should be updated by the user
        top_k_matches=2,
        threshold_distance=1.5
    )
    config_data.llm_config = OpenAiLlmConfigData(
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        completion_model='gpt-4',
        temperature=0.5
    )
    config_data.embedding_config = OpenAiEmbeddingConfigData(
        model_name='text-embedding-ada-002',
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
    )
    answer_generator = AnswerGeneratorService(config_data)
    result = answer_generator.generate_answer(request_data)
    assert result is not None and result.generated_answer is not None


def test_generate_answer_opensource_rag():
    request_data = AnswerGeneratorRequestData(
        # Query should be updated by the user
        query="",
        # Embedding type should be updated by the user
        embedding_type="opensource",
        # Rag should be updated by the user
        rag=True
    )
    config_data = AnswerGeneratorConfigData(
        # Vector_db_directory should be updated by the user
        vector_db_directory="C:/test/vectordb",
        context_length=1000,
        # Top_k_matches should be updated by the user
        top_k_matches=5,
        threshold_distance=1.5
    )
    config_data.llm_config = OpenSourceLlmConfigData(
        # Steps to download a model can be referred from README.md file.model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/roberta-base-squad2"
    )
    config_data.embedding_config = OpenSourceEmbeddingConfigData(
        # Steps to download a model can be referred from README.md file.model_path below needs to be updated by the user accordingly.
        model_path='C:/MyProgramFiles/AI/models/all-MiniLM-L6-v2'
    )
    answer_generator = AnswerGeneratorService(config_data)
    result = answer_generator .generate_answer(request_data)
    assert result is not None and result.generated_answer is not None


def test_generate_answer_openai():
    request_data = AnswerGeneratorRequestData(
        # Query should be updated by the user
        query="",
        # Embedding type should be updated by the user
        embedding_type="openai",
        # Rag should be updated by the user
        rag=False
    )
    config_data = AnswerGeneratorConfigData(
        llm_config=OpenAiLlmConfigData(
            open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
            open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
            open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
            completion_model='gpt-4',
            temperature=0.5
        ))
    answer_generator = AnswerGeneratorService(config_data)
    result = answer_generator.generate_answer(request_data)
    assert result is not None and result.generated_answer is not None


def test_generate_answer_with_no_embedding_type():
    request_data = AnswerGeneratorRequestData(
        # Query should be updated by the user
        query="what is the capital of India?",
        # Rag should be updated by the user
        rag=False
    )
    config_data = AnswerGeneratorConfigData(
        # Vector_db_directory should be updated by the user
        vector_db_directory="C:/test/vectordb",
        context_length=1000,
        # Top_k_matches should be updated by the user
        top_k_matches=1,
        threshold_distance=1.5
    )
    with pytest.raises(ValueError):
        answer_generator = AnswerGeneratorService(config_data)
        answer_generator.generate_answer(request_data)

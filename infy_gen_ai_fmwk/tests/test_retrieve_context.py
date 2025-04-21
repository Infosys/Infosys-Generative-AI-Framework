# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.service.context_retriever_service import ContextRetrieverService
from infy_gen_ai_fmwk.data.config_data import ContextRetrieverConfigData, OpenAiEmbeddingConfigData, OpenSourceEmbeddingConfigData
from infy_gen_ai_fmwk.data.request_data import ContextRetrieverRequestData


def test_retrieve_context_openai():
    request_data = ContextRetrieverRequestData(
        # Query should be updated by the user
        query="",
        # Embedding type should be updated by the user
        embedding_type="openai"
    )
    config_data = ContextRetrieverConfigData(
        # Vector db directory file path to be updated by the user
        vector_db_directory="C:/test/vectordb",
        context_length=1000,
        # Top kmatches should be updated by the user
        top_k_matches=1,
        threshold_distance=1.5
    )
    config_data.embedding_config = OpenAiEmbeddingConfigData(
        model_name='text-embedding-ada-002',
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
    )
    context_retriever = ContextRetrieverService(config_data)
    result = context_retriever.retrieve_context(request_data)
    assert result is not None and result.closest_matches is not None


def test_retrieve_context_opensource():
    request_data = ContextRetrieverRequestData(
        # Query should be updated by the user
        query="",
        embedding_type="opensource"
    )
    config_data = ContextRetrieverConfigData(
        # Vector db directory file path to be updated by the user
        vector_db_directory="C:/test/vectordb",
        context_length=1000,
        # Top k matches should be updated by the user
        top_k_matches=5,
        threshold_distance=1.5
    )
    config_data.embedding_config = OpenSourceEmbeddingConfigData(
        # Steps to download a model can be referred from README.md file.model_path below needs to be updated by the user accordingly.
        model_path='C:/MyProgramFiles/AI/models/all-MiniLM-L6-v2'
    )
    context_retriever = ContextRetrieverService(config_data)
    result = context_retriever.retrieve_context(request_data)
    assert result is not None and result.closest_matches is not None


def test_retrieve_context_with_no_embedding_type():
    request_data = ContextRetrieverRequestData(
        # Query should be updated by the user
        query=""
    )
    config_data = ContextRetrieverConfigData(
        # Vector db directory file path to be updated by the user
        vector_db_directory="C:/test/vectordb",
        context_length=1000,
        # Top k matches should be updated by the user
        top_k_matches=5,
        threshold_distance=1.5
    )
    context_retriever = ContextRetrieverService(config_data)
    with pytest.raises(ValueError):
        context_retriever.retrieve_context(request_data)

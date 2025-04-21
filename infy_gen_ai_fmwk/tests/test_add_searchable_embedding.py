# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.service.add_searchable_embeddings import AddSearchableEmbeddingsService
from infy_gen_ai_fmwk.data.config_data import AddSearchableEmbeddingsConfigData, OpenAiEmbeddingConfigData, OpenSourceEmbeddingConfigData
from infy_gen_ai_fmwk.data.request_data import AddSearchableEmbeddingsRequestData


def test_add_searchable_embeddings_valid_file_openai():
    # Input file path should be updated by the user
    test_file_path = []
    request_data = AddSearchableEmbeddingsRequestData(
        file_paths=test_file_path,
        # Embedding type should be updated by the user
        embedding_type='openai'
    )
    config_data = AddSearchableEmbeddingsConfigData(
        token_group_size=2048,
        chunk_length=800,
        # Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
        vocab_dir_path="C:/MyProgramFiles/AI/models/gpt2",
        # Vector db directory file path to be updated by the user
        vector_db_directory="C:/test/vectordb"
    )
    config_data.embedding_config = OpenAiEmbeddingConfigData(
        model_name='text-embedding-ada-002',
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'])
    search_service = AddSearchableEmbeddingsService(config_data)
    result = search_service.add_searchable_embeddings(request_data)
    assert result is not None and result.content is not None


def test_add_searchable_embeddings_valid_file_opensource():
    # Input file path should be updated by the user
    test_file_path = []
    request_data = AddSearchableEmbeddingsRequestData(
        file_paths=test_file_path,
        # Embedding type should be updated by the user
        embedding_type='opensource'
    )
    config_data = AddSearchableEmbeddingsConfigData(
        token_group_size=2048,
        chunk_length=800,
        # Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
        vocab_dir_path="C:/MyProgramFiles/AI/models/gpt2",
        # Vector db directory file path to be updated by the user
        vector_db_directory="C:/test/vectordb"
    )
    config_data.embedding_config = OpenSourceEmbeddingConfigData(
        # Steps to download a model can be referred from README.md file.model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/all-MiniLM-L6-v2"
    )
    search_service = AddSearchableEmbeddingsService(config_data)
    result = search_service.add_searchable_embeddings(request_data)
    assert result is not None and result.content is not None


def test_add_searchable_embeddings_with_invalid_and_valid_files():
    # Input file path having valid and invalid extension can be updated by the user
    test_file_path = [" ","data/sample/input/QR_Q1_2025.mp3"]
    request_data = AddSearchableEmbeddingsRequestData(
        file_paths=test_file_path,
        # Embedding type should be updated by the user
        embedding_type='opensource'
    )
    config_data = AddSearchableEmbeddingsConfigData(
        token_group_size=2048,
        chunk_length=800,
        # Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
        vocab_dir_path="C:/MyProgramFiles/AI/models/gpt2",
        # Vector db directory file path to be updated by the user
        vector_db_directory="C:/test/vectordb"
    )
    config_data.embedding_config = OpenSourceEmbeddingConfigData(
        # Steps to download a model can be referred from README.md file.model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/all-MiniLM-L6-v2"
    )
    search_service = AddSearchableEmbeddingsService(config_data)
    result = search_service.add_searchable_embeddings(request_data)
    assert result is not None and result.content is not None and result.rejected_files is not None


def test_add_searchable_embeddings_with_invalid_files_opensource():
    # Input file path having invalid extension can be updated by the user
    test_file_path = ["data/sample/input/QR_Q1_2025.txt","data/sample/input/QR_Q1_2025.mp3"]
    request_data = AddSearchableEmbeddingsRequestData(
        file_paths=test_file_path,
        # Embedding type should be updated by the user
        embedding_type='opensource'
    )
    config_data = AddSearchableEmbeddingsConfigData(
        token_group_size=2048,
        chunk_length=800,
        # Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
        vocab_dir_path="C:/MyProgramFiles/AI/models/gpt2",
        # Vector db directory file path to be updated by the user
        vector_db_directory="C:/test/vectordb"
    )
    config_data.embedding_config = OpenSourceEmbeddingConfigData(
        # Steps to download a model can be referred from README.md file.model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/all-MiniLM-L6-v2"
    )
    search_service = AddSearchableEmbeddingsService(config_data)
    with pytest.raises(ValueError):
        search_service.add_searchable_embeddings(request_data)


def test_add_searchable_embeddings_with_no_embedding_type():
    # Input file path should be updated by the user
    test_file_path = []
    request_data = AddSearchableEmbeddingsRequestData()
    request_data.file_paths = test_file_path
    config_data = AddSearchableEmbeddingsConfigData(
        token_group_size=2048,
        chunk_length=800,
        # Steps to download a model can be referred from README.md file.The vocab_dir_path below needs to be updated by the user accordingly.
        vocab_dir_path="C:/MyProgramFiles/AI/models/gpt2",
        # Vector db directory file path to be updated by the user
        vector_db_directory="C:/test/vectordb"
    )
    search_service = AddSearchableEmbeddingsService(config_data)
    with pytest.raises(ValueError):
        search_service.add_searchable_embeddings(request_data)

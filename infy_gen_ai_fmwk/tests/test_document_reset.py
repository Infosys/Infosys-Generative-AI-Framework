# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.data.config_data import DocumentResetterConfigData
from infy_gen_ai_fmwk.service.document_reset import DocumentResetService


def test_reset_document_openai():
    config_data = DocumentResetterConfigData()
    # Vector db directory file path to be updated by the user
    config_data.vector_db_directory = "C:/test/vectordb"
    # Collection name to be updated by the user
    config_data.collection_name = 'openai_embedding_collection'
    document_resetter = DocumentResetService(config_data)
    result = document_resetter.reset_document()
    assert result is not None and result.content is not None


def test_reset_document_opensource():
    config_data = DocumentResetterConfigData()
    # Vector db directory file path to be updated by the user
    config_data.vector_db_directory = "C:/test/vectordb"
    # Collection name to be updated by the user
    config_data.collection_name = 'opensource_embedding_collection'
    document_resetter = DocumentResetService(config_data)
    result = document_resetter.reset_document()
    assert result is not None and result.content is not None


def test_reset_document_invalid_collection_name():
    config_data = DocumentResetterConfigData()
    # Vector db directory file path to be updated by the user
    config_data.vector_db_directory = "C:/test/vectordb"
    # Collection name to be updated by the user
    config_data.collection_name = 'open_embedding_collection'
    with pytest.raises(ValueError):
        document_resetter = DocumentResetService(config_data)
        document_resetter.reset_document()

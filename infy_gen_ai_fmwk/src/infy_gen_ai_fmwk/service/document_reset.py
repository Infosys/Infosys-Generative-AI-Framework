# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#
"""This file contains the service class for resetting document"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.data.config_data import DocumentResetterConfigData
from infy_gen_ai_fmwk.service.provider.embedding_provider import EmbeddingProvider
from infy_gen_ai_fmwk.data.response_data import DocumentResetterResponseData

if not os.path.exists('logs'):
    os.makedirs('logs')
logger = logging.getLogger('logger')
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = TimedRotatingFileHandler(
        'logs/gen_ai_fmwk.log', when='midnight', interval=1)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DocumentResetService:
    def __init__(self, config_data: DocumentResetterConfigData):
        self.config_data = config_data
        logger.debug('DocumentResetService initialized')

    def reset_document(self):
        response_data = DocumentResetterResponseData()
        try:
            embedding_provider = EmbeddingProvider(self.config_data)
            embedding_provider.reset_dbs(self.config_data)
            content = 'Document reset'
            response_data.content = content
            return response_data
        except ValueError as e:
            logger.error(f"Exception: {str(e)}")
            raise ValueError(str(e))
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise Exception(str(e))

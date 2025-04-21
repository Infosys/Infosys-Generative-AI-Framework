# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#
"""This file contains the service class for retrieving context"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.data.config_data import ContextRetrieverConfigData
from infy_gen_ai_fmwk.data.request_data import ContextRetrieverRequestData, ModelType
from infy_gen_ai_fmwk.data.response_data import ContextRetrieverResponseData
from infy_gen_ai_fmwk.service.provider.retriever_provider import RetrieverProvider

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


class ContextRetrieverService:
    def __init__(self, config_data: ContextRetrieverConfigData):
        self.config_data = config_data
        logger.info('ContextRetrieverService initialized')

    def retrieve_context(self, request_data: ContextRetrieverRequestData) -> ContextRetrieverResponseData:
        response_data = ContextRetrieverResponseData()
        try:
            if not request_data.embedding_type:
                request_data.embedding_type = ModelType.OPENAI
                logger.error(
                    FileConstants.get_invalid_model_type_error_msg())
                raise ValueError(
                    FileConstants.get_invalid_model_type_error_msg())
            context_retriever = RetrieverProvider()
            closest_matches = context_retriever.retrieve_context(
                request_data.query, request_data.embedding_type, self.config_data)
            response_data.closest_matches = closest_matches
            logger.info('Retrieved context')
            return response_data
        except ValueError as e:
            raise e
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise Exception(str(e))

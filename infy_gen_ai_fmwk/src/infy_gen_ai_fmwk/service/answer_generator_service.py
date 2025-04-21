# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the AnswerGeneratorService class which is responsible for generating answers for the given query."""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.data.config_data import AnswerGeneratorConfigData
from infy_gen_ai_fmwk.data.request_data import AnswerGeneratorRequestData, ModelType
from infy_gen_ai_fmwk.data.response_data import AnswerGeneratorResponseData
from infy_gen_ai_fmwk.service.provider.generate_answer_provider import GenerateAnswerProvider

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


class AnswerGeneratorService:
    def __init__(self, config_data: AnswerGeneratorConfigData):
        self.config_data = config_data
        logger.info('AnswerGeneratorService initialized')

    def generate_answer(self, request_data: AnswerGeneratorRequestData) -> AnswerGeneratorResponseData:
        response_data = AnswerGeneratorResponseData()
        try:
            if not request_data.embedding_type:
                request_data.embedding_type = ModelType.OPENAI
                logger.error(
                    FileConstants.get_invalid_model_type_error_msg())
                raise ValueError(
                    FileConstants.get_invalid_model_type_error_msg())
            rag = request_data.rag
            answer_generator = GenerateAnswerProvider(rag=rag)
            answers = answer_generator.generate_answer(self.config_data,
                                                       query=request_data.query, type=request_data.embedding_type)
            response_data.generated_answer = answers
            logger.info('Generated answer %s', answers)
            return response_data
        except ValueError as e:
            raise e
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise Exception(str(e))

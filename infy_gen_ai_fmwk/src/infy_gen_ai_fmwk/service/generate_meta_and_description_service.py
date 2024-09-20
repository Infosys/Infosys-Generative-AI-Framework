# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for generating meta description from text file"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.data.request_data import MetaAndDescriptionGeneratorRequestData, MetaAndDescriptionGeneratorResponseData
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.service.provider.openai import OpenAIAPI
if not os.path.exists('logs'):
    os.makedirs('logs')
logger = logging.getLogger('logger')
logger.setLevel(logging.ERROR)
handler = TimedRotatingFileHandler(
    'logs/gen_ai_fmwk.log', when='midnight', interval=1)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class GenerateMetaAndDescriptionService:
    def __init__(self, config_data: OpenAiLlmConfigData):
        self.config_data = config_data
        logger.info('GenerateMetaAndDescriptionService initialized')

    def get_meta_description(self, request_data: MetaAndDescriptionGeneratorRequestData) -> MetaAndDescriptionGeneratorResponseData:
        response_data = MetaAndDescriptionGeneratorResponseData()
        if not Utils.check_extension(request_data.file_path, FileConstants.VALID_EXTENSIONS['txt_file']):
            logger.error(
                FileConstants.get_invalid_file_error_msg('txt_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('txt_file'))
        try:
            contents = Utils.readTxt(file_path=request_data.file_path)
            logger.info('Read file')
            answer = self.__get_metadata(contents)
            response_data.meta_descr_response = answer
            logger.info('Generated meta description')
            return response_data

        except Exception as e:
            logger.error('Error generating meta description')
            raise ValueError(str(e))

    def __get_metadata(self, content):
        api = OpenAIAPI()
        llm_config = self.config_data
        keyword_prompt = llm_config.prompt_template.format(content, "", "")
        keyword_responses = api.get_chat_completion(
            keyword_prompt, engine=llm_config.completion_model, temperature=llm_config.temperature, stop=['Meta Description:'])
        keyword_response = keyword_responses[0]
        logger.info('Generated keywords')

        desc_prompt = llm_config. prompt_template.format(
            content, keyword_response, content)
        desc_responses = api.get_chat_completion(
            desc_prompt, engine=llm_config.completion_model, temperature=llm_config.temperature)
        desc_response = desc_responses[0]
        logger.info('Generated description')

        result = {'keywords': keyword_response,
                  'description': desc_response}

        return result

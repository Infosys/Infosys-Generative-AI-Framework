# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for generating unit tests"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import UnitTestCreatorRequestData, UnitTestCreatorResponseData
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


class CreateUnitTestsService:
    def __init__(self, config_data: OpenAiLlmConfigData):
        self.config_data = config_data
        logger.info('CreateUnitTestsService initialized')

    def create_unit_tests(self, request_data: UnitTestCreatorRequestData) -> UnitTestCreatorResponseData:
        response_data = UnitTestCreatorResponseData()
        if not Utils.check_extension(request_data.src_code_file_path, FileConstants.VALID_EXTENSIONS['source_file']):
            logger.error(
                FileConstants.get_invalid_file_error_msg('source_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('source_file'))
        contents = Utils.readTxt(file_path=request_data.src_code_file_path)
        logger.info('Read source code file')
        lang = request_data.lang
        try:
            answer = self.__to_generate_test_cases(contents, lang)
            response_data.unit_test_response = answer
            logger.info('Generated unit tests')
            return response_data
        except Exception as e:
            logger.error('Error generating unit tests')
            raise Exception(str(e))

    def __to_generate_test_cases(self, code: str, lang: str):
        llm_config = self.config_data
        prompt = llm_config.prompt_template.format(lang=lang)

        prompt += f'{code}\nTest Cases:'
        api = OpenAIAPI()
        return api.get_chat_completion(prompt, temperature=llm_config.temperature, engine=llm_config.completion_model, stop=["6."])[0]

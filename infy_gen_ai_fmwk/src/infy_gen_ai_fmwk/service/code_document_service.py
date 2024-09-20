# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for generating code documentation"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import CodeDocumentRequestData, CodeDocumentResponseData
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


class CodeDocumentService:
    def __init__(self, config_data: OpenAiLlmConfigData):
        self.config_data = config_data
        logger.info('CodeDocumentService initialized')

    def generate_code_docs(self, request_data: CodeDocumentRequestData) -> CodeDocumentResponseData:
        response_data = CodeDocumentResponseData()
        if not Utils.check_extension(request_data.src_code_file_path, FileConstants.VALID_EXTENSIONS['source_file']):
            logger.error(
                FileConstants.get_invalid_file_error_msg('source_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('source_file'))
        contents = Utils.readTxt(file_path=request_data.src_code_file_path)
        logger.info('Read source code file')
        lang = request_data.lang
        try:

            api = OpenAIAPI()
            llm_config = self.config_data
            prompt = llm_config.prompt_template.format(
                lang=lang, contents=contents)

            answer = api.get_chat_completion(
                prompt, temperature=llm_config.temperature, engine=llm_config.completion_model, stop=['Regenerated Code:', 'Docs:'])
            response_data.code_document_response = answer[0]
            logger.info('Generated code documentation')
            return response_data
        except Exception as e:
            logger.error('Error generating code documentation')
            raise Exception(str(e)) 

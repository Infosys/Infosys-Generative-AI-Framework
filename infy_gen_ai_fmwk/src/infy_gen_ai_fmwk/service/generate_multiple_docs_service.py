# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class to generate multiple code documentation"""
import os
import zipfile
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import MultipleDocsGeneratorRequestData, MultipleDocsGeneratorResponseData
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


class GenerateMultipleDocsService:
    def __init__(self, config_data: OpenAiLlmConfigData):
        self.config_data = config_data
        logger.info('GenerateMultipleDocsService initialized')

    def generate_multi_docs(self, request_data: MultipleDocsGeneratorRequestData) -> MultipleDocsGeneratorResponseData:
        response_data = MultipleDocsGeneratorResponseData()
        rejected_files = []

        for file_path in request_data.src_code_file_paths:
            if not Utils.check_extension(file_path, FileConstants.VALID_EXTENSIONS['source_file']):
                rejected_files.append(file_path)

        if len(request_data.src_code_file_paths) == len(rejected_files):
            logger.error(
                FileConstants.get_invalid_file_error_msg('source_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('source_file'))
        try:
            os.makedirs(os.path.dirname(request_data.output_compressed_file_path), exist_ok=True)
            if not request_data.output_compressed_file_path.lower().endswith('.zip'):
                request_data.output_compressed_file_path += '.zip'
            with zipfile.ZipFile(request_data.output_compressed_file_path, 'w') as zip:
                for file_path in request_data.src_code_file_paths:
                    if file_path in rejected_files:
                        continue

                    contents = Utils.readTxt(file_path=file_path)
                    logger.info('Read source code file')
                    llm_config = self.config_data
                    api = OpenAIAPI()
                    prompt = llm_config.prompt_template.format(
                        contents=contents)

                    code_docs = api.get_chat_completion(
                        prompt, temperature=llm_config.temperature, engine=llm_config.completion_model, stop=['Regenerated Code:', 'Docs:'])
                    answer = code_docs[0]
                    filename_without_extension = os.path.splitext(
                        os.path.basename(file_path))[0]
                    zip.writestr(filename_without_extension + '.txt', answer)
            response_data.code_docs_response = request_data.output_compressed_file_path
            response_data.rejected_files = rejected_files
            logger.info('Generated multiple code documentation')
            return response_data
        except Exception as e:
            logger.error('Error generating multiple code documentation')
            raise Exception(str(e))

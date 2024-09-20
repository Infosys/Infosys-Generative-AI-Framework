# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for translating code"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.service.provider.openai import OpenAIAPI
from infy_gen_ai_fmwk.data.config_data import CodeTranslatorConfigData
from infy_gen_ai_fmwk.data.request_data import CodeTranslatorRequestData, CodeTranslatorResponseData
from infy_gen_ai_fmwk.common.tokenizer import Tokenizer
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


class CodeTranslatorService:
    def __init__(self, config_data: CodeTranslatorConfigData):
        self.config_data = config_data
        self.tokenizer = Tokenizer(
            self.config_data.vocab_dir_path, self.config_data)
        logger.info('CodeTranslatorService initialized')

    def translate_code(self, request_data: CodeTranslatorRequestData) -> CodeTranslatorResponseData:
        response_data = CodeTranslatorResponseData()
        if not Utils.check_extension(request_data.src_code_file_path, FileConstants.VALID_EXTENSIONS['source_file']):
            logger.error(
                FileConstants.get_invalid_file_error_msg('source_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('source_file'))
        contents = Utils.readTxt(file_path=request_data.src_code_file_path)
        logger.info('Read source code file')
        src_lang = request_data.src_lang
        tgt_lang = request_data.tgt_lang
        try:
            answer = self.__get_translated_code(
                contents, src_lang, tgt_lang)
            response_data.translated_code = answer
            logger.info('Generated translated code')
            return response_data

        except Exception as e:
            logger.error('Error generating translated code')
            raise Exception(str(e)) 

    def __get_translated_code(self, source_code, source_lang, target_lang):

        llm_config = self.config_data.open_ai_config
        num_tokens = len(
            self.tokenizer.tokenize(source_code))
        print(self.config_data.token_limit)
        if (num_tokens > self.config_data.token_limit):
            logger.info('Token limit reached. Either decrease size of source code or remove a few examples')

        prompt = self.__to_generate_prompt(
            source_code, source_lang, target_lang)
        api = OpenAIAPI()
        answers = api.get_chat_completion(
            prompt, temperature=llm_config.temperature, engine=llm_config.completion_model, stop=[f'{source_lang}:'])
        return answers[0]

    def __to_generate_prompt(self, src_code, src_lang, tgt_lang):
        llm_config = self.config_data.open_ai_config
        prompt = llm_config.prompt_template.format(
            src_lang=src_lang, tgt_lang=tgt_lang)

        prompt += f'{src_lang}:\n{src_code}\n'
        prompt += f'{tgt_lang}:\n'

        return prompt

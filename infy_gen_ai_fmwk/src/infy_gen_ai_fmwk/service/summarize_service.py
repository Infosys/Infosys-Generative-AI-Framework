# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for summarizing PDF file"""
import os
import math
import logging
import traceback
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.pypdf import read_pdf_from_file
from infy_gen_ai_fmwk.common.tokenizer import Tokenizer
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.data.config_data import SummarizerConfigData
from infy_gen_ai_fmwk.data.request_data import SummarizerRequestData, SummarizerResponseData
from infy_gen_ai_fmwk.service.provider.openai import OpenAIAPI
from enum import Enum
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

api = OpenAIAPI()

class SummarizationType(Enum):
    LINEAR = 'linear'
    RECURSIVE = 'recursive'


class SummarizeService:
    def __init__(self, config_data: SummarizerConfigData):
        self.config_data = config_data
        self.tokenizer = Tokenizer(
            self.config_data.vocab_dir_path, self.config_data)
        logger.info('SummarizeService initialized')

    def set_summarization_type(self, summarization_type):
        if summarization_type not in SummarizationType.__members__:
            raise ValueError(
                f'Invalid summarization type. Expected one of {list(SummarizationType.__members__.keys())}')
        self.summarization_type = SummarizationType[summarization_type]

    def summarize(self, request_data: SummarizerRequestData) -> SummarizerResponseData:
        response_data = SummarizerResponseData()
        if not Utils.check_extension(request_data.file_path, FileConstants.VALID_EXTENSIONS['pdf_file']):
            logger.error(FileConstants.get_invalid_file_error_msg('pdf_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('pdf_file'))
        content = request_data.file_path
        logger.info('Read PDF file')
        try:
            answer = self.__to_summarize_pdf(
                content, request_data.summarization_type)
            response_data.summary = answer
            logger.info('Generated summary')
            return response_data
        except Exception as e:
            logger.error('Error generating summary')
            logger.error(traceback.format_exc())
            raise Exception(str(e))

    def __to_summarize_pdf(self, file, summarization_type):
        if summarization_type == 'recursive':
            text_chunks = self.tokenizer.tokenize_pdf_from_file(
                file)
            logger.info(
                f'Tokenizing done. {len(text_chunks)} text chunks are generated')
            summary = self.__to_group_and_summarize_recursive(text_chunks)
        elif summarization_type == 'linear':
            page_chunks = read_pdf_from_file(file)
            logger.info(
                f'{len(page_chunks)} text chunks are generated')

            logger.info('Starting summarization..')
            summary = self.__to_group_and_summarize_linear(
                ''.join(page_chunks))

        return summary

    def __to_group_and_summarize_recursive(self, text_chunks):
        llm_config = self.config_data.open_ai_config
        if len(text_chunks) == 1:
            summaries = api.get_chat_completion(
                f'Summarize the following in 5-10 sentences : \n {text_chunks[0]}', engine=llm_config.completion_model, temperature=llm_config.temperature)
            return summaries[0]

        summary_chunks = [api.get_chat_completion(
            f'Summarize the following in 10-15 sentences: \n {x}', engine=llm_config.completion_model, temperature=llm_config.temperature)[0] for x in text_chunks]
        group_size = int(math.floor(
            self.config_data.token_group_size/self.config_data.summary_tokens))
        index = 0
        new_groups = list()
        while True:
            if index + group_size >= len(summary_chunks):
                new_groups.append(' \n\n'.join(
                    [x for x in summary_chunks[index:]]))
                break
            else:
                new_groups.append(' \n\n'.join(
                    [x for x in summary_chunks[index: index + group_size]]))
                index += group_size

        return self.__to_group_and_summarize_recursive(new_groups)

    def __to_group_and_summarize_linear(self, text):

        group = []
        groups = []

        blocks = text.split("\n")
        for block in blocks:
            if len((" ").join(group).split(" ")) + len(block.split(" ")) > self.config_data.word_limit:
                groups.append(group)
                group = [block]
            else:
                group.append(block)

        if len(group) > 0:
            groups.append(group)

        tldrs = []
        for group in groups:
            text = ("\n").join(group)
            llm_config = self.config_data.open_ai_config
            prompt = llm_config.prompt_template.format(text=text)
            tldr = api.get_chat_completion(
                prompt, temperature=llm_config.temperature, engine=llm_config.completion_model)
            tldrs.append(tldr[0])

        return ' '.join(tldrs)

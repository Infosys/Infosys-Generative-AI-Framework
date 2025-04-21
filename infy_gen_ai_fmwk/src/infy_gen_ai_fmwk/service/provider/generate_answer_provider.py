# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.prompt_template import PromptTemplate
from infy_gen_ai_fmwk.data.request_data import ModelType
from infy_gen_ai_fmwk.service.provider.llm_api import LLMAPI
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


class GenerateAnswerProvider:
    def __init__(self, rag: bool = False):
        self.api_obj = LLMAPI()
        self.rag = rag
        self.last_query_answer = None
        self.context = RetrieverProvider()

    def generate_answer(self, config_data, query, type, completions=1):
        if self.rag:
            prompt = PromptTemplate.GENERATE_ANSWER_PROMPT
            context = self.context.retrieve_context(
                query, type, config_data)
            prompt += f'Context:\n{context}\n'
            prompt += f'Question : {query}\n'
            prompt += 'Answer :'
        else:
            prompt = PromptTemplate.GENERATE_ANSWER_PROMPT
            prompt += f'Question : {query}\n'
            prompt += 'Answer :'
        logger.info('query: %s', query)
        if type == ModelType.OPENAI:
            answers = self.api_obj._get_completion(
                type, prompt=prompt, completions=completions, temperature=config_data.llm_config.temperature, engine=config_data.llm_config.completion_model)
        elif type == ModelType.OPENSOURCE:
            answers = self.api_obj._get_completion(
                type, prompt=prompt, completions=completions, model_mapping=config_data.llm_config.model_path)
        self.last_query_answer = answers[0]
        return answers

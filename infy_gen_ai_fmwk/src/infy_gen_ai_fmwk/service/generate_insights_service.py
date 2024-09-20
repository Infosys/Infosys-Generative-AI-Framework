# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class to generate insights from the given data"""
import io
import os
import pandas as pd
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import InsightsGeneratorRequestData, InsightsGeneratorResponseData
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


class GenerateInsights:
    def __init__(self, config_data: OpenAiLlmConfigData):
        self.config_data = config_data
        logger.info('GenerateInsights initialized')

    def generate_insights(self, request_data: InsightsGeneratorRequestData) -> InsightsGeneratorResponseData:

        response_data = InsightsGeneratorResponseData()
        if not Utils.check_extension(request_data.file_path, FileConstants.VALID_EXTENSIONS['excel_file']):
            logger.error(
                FileConstants.get_invalid_file_error_msg('excel_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('excel_file'))

        with open(request_data.file_path, 'rb') as f:
            contents = f.read()
            logger.info('Read file')
        try:
            if request_data.file_path.endswith('.csv'):
                data = pd.read_csv(io.BytesIO(contents))
            else:
                data = pd.read_excel(io.BytesIO(contents))
            data = self.__to_get_content_from_dataframe(data)
            llm_config = self.config_data
            api = OpenAIAPI()
            prompt = llm_config.prompt_template.format(data=data)
            answer = api.get_chat_completion(
                prompt, temperature=llm_config.temperature, engine=llm_config.completion_model, completions=request_data.completions)
            response_data.insights_response = answer
            logger.info('Generated insights')
            return response_data
        except Exception as e:
            logger.error('Error generating insights')
            raise Exception(str(e))

    def __to_get_content_from_dataframe(self, dataframe):
        data = ''
        for _, row in dataframe.iterrows():
            for column in dataframe.columns:
                data += f'{row[column]}|'
            data += '\n'

        return data

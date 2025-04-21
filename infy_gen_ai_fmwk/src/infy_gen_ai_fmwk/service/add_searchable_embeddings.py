# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for adding searchable embeddings"""
import os
import pandas as pd
import numpy as np
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.data.request_data import AddSearchableEmbeddingsRequestData, ModelType
from infy_gen_ai_fmwk.data.response_data import AddSearchableEmbeddingsResponseData
from infy_gen_ai_fmwk.data.config_data import AddSearchableEmbeddingsConfigData
from infy_gen_ai_fmwk.service.provider.embedding_provider import EmbeddingProvider
from infy_gen_ai_fmwk.service.provider.openai import OpenAIAPI
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.common.tokenizer import Tokenizer
from infy_gen_ai_fmwk.service.provider.open_source import OpenSourceAPI

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


class AddSearchableEmbeddingsService:
    def __init__(self, config_data: AddSearchableEmbeddingsConfigData):
        self.config_data = config_data
        self.tokenizer = Tokenizer(
            self.config_data.vocab_dir_path, self.config_data)
        logger.info('AddSearchableEmbeddingsService initialized')

    def add_searchable_embeddings(self, request_data: AddSearchableEmbeddingsRequestData) -> AddSearchableEmbeddingsResponseData:
        response_data = AddSearchableEmbeddingsResponseData()
        rejected_files = []
        for file_path in request_data.file_paths:
            if not Utils.check_extension(file_path, FileConstants.VALID_EXTENSIONS['pdf_file']):
                rejected_files.append(file_path)
            if not request_data.embedding_type:
                request_data.embedding_type = ModelType.OPENAI
                logger.error(
                    FileConstants.get_invalid_model_type_error_msg())
                raise ValueError(
                    FileConstants.get_invalid_model_type_error_msg())
            if len(request_data.file_paths) == len(rejected_files):
                logger.error(
                    FileConstants.get_invalid_file_error_msg('pdf_file'))
                raise ValueError(
                    FileConstants.get_invalid_file_error_msg('pdf_file'))
        try:
            for file_path in request_data.file_paths:
                if file_path in rejected_files:
                    continue
                with open(file_path, 'rb') as file:
                    openai_embed_df, opensource_embed_df = self.__create_doc_embeddings_from_file(
                        file, request_data.embedding_type)
                    embedding_provider = EmbeddingProvider(self.config_data)
                    if request_data.embedding_type == ModelType.OPENAI:
                        embedding_provider.add_embeddings(
                            ModelType.OPENAI, openai_embed_df, file_path)
                    elif request_data.embedding_type == ModelType.OPENSOURCE:
                        embedding_provider.add_embeddings(
                            ModelType.OPENSOURCE, opensource_embed_df, file_path)
            content = 'Embeddings have been added to the searchable database'
            response_data.content = content
            response_data.rejected_files = rejected_files
            logger.info(
                'Embeddings have been added to the searchable database')
            return response_data
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise Exception(str(e))

    def __create_doc_embeddings_from_file(self, file, type):
        openai_embed_df = pd.DataFrame()
        opensource_embed_df = pd.DataFrame()
        openai_file_df, opensource_file_df = self.__process_pdf_from_file(
            file, type)
        openai_embed_df = pd.concat(
            [openai_embed_df, openai_file_df], ignore_index=True)
        opensource_embed_df = pd.concat(
            [opensource_embed_df, opensource_file_df], ignore_index=True)
        return openai_embed_df, opensource_embed_df

    def __process_pdf_from_file(self, file, type):
        openai_embed_df = pd.DataFrame()
        opensource_embed_df = pd.DataFrame()
        text_chunks = self.tokenizer.tokenize_pdf_from_file(file)
        openai_embed_dict, opensource_embed_dict = self.__process_doc(
            text_chunks, type)
        for text, embed in openai_embed_dict.items():
            new_row = pd.DataFrame(
                [{'text': text, 'embeddings': np.array(embed)}])
            openai_embed_df = pd.concat(
                [openai_embed_df, new_row], ignore_index=True)
        for text, embed in opensource_embed_dict.items():
            new_row = pd.DataFrame(
                [{'text': text, 'embeddings': np.array(embed)}])
            opensource_embed_df = pd.concat(
                [opensource_embed_df, new_row], ignore_index=True)
        return openai_embed_df, opensource_embed_df

    def __process_doc(self, text_chunks, type):
        openai_embed_dict = dict()
        opensource_embed_dict = dict()
        for text in text_chunks:
            if type == ModelType.OPENAI:
                api = OpenAIAPI()
                openai_embed_dict[text] = api.get_embedding(
                    text, engine=self.config_data.embedding_config.model_name)
            elif type == ModelType.OPENSOURCE:
                opensource_api = OpenSourceAPI()
                opensource_embed_dict[text] = opensource_api.get_embedding(
                    text, engine=self.config_data.embedding_config.model_path)
        return openai_embed_dict, opensource_embed_dict

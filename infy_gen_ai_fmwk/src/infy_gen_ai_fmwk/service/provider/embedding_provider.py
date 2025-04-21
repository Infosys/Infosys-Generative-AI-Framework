# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import logging
from logging.handlers import TimedRotatingFileHandler
import os
import pandas as pd
from infy_gen_ai_fmwk.data.request_data import ModelType
from infy_gen_ai_fmwk.service.provider.chroma_db import TransformersVectorDB

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


class EmbeddingProvider:
    def __init__(self, config_data):
        self.vector_db = TransformersVectorDB(config_data)

    def add_embeddings(self, type, embed_df: pd.DataFrame, metadata_string: str):
        if type == ModelType.OPENAI:
            collection_name = 'openai_embedding_collection'
        elif type == ModelType.OPENSOURCE:
            collection_name = 'opensource_embedding_collection'
        logger.info("Collection Name: %s", collection_name)
        self.vector_db.add_embeddings_to_collection(
            embed_df, metadata_string, collection_name)
        
    def reset_dbs(self, config_data):
        self.vector_db.reset_collection(config_data.collection_name)    

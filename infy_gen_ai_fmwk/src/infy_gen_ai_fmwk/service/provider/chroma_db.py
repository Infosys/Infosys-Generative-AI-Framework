# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import logging
from logging.handlers import TimedRotatingFileHandler
import uuid
import pandas as pd
import chromadb
from chromadb.config import Settings

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


class TransformersVectorDB:
    def __init__(self, config_data):
        self.client = chromadb.Client(
            Settings(chroma_db_impl='duckdb+parquet', persist_directory=config_data.vector_db_directory))
        self.collection = None

    def load_collection(self, name):
        self.collection = self.client.get_or_create_collection(name=name)
        logger.info(
            f'Loaded collection : {name} with {self.collection.count()} records')

    def add_embeddings_to_collection(self, embeddings_df: pd.DataFrame, metadata_string: str, collection_name):
        self.load_collection(collection_name)
        self.collection.add(
            documents=list(embeddings_df['text']),
            embeddings=[x.tolist() for x in embeddings_df['embeddings']],
            metadatas=[{'doc': max([metadata_string, ''], key=len)}
                       for _ in range(len(embeddings_df))],
            ids=[str(uuid.uuid4()) for _ in range(len(embeddings_df))]
        )
        self.client.persist()
        logger.info(
            f'Sucessfully added docs to collection. New collection size is : {self.collection.count()}')

    def query_collection(self, query_embeddings: list, collection_name):
        self.load_collection(collection_name)
        if self.collection.count() == 0:
            return {'documents': []}
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=self.collection.count() if self.collection.count() < 5 else 5
        )
        return results

    def reset_collection(self, collection_name: str):
        try:
            self.client.get_collection(collection_name)
            self.client.delete_collection(name=collection_name)
            self.load_collection(collection_name)
            logger.info(f'Collection {collection_name} reset successfully')
        except ValueError as e:
            logger.error(
                f'The collection {collection_name} does not exist in the DB')
            raise ValueError(str(e))

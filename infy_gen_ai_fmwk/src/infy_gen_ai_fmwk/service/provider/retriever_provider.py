# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import json
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.data.request_data import ModelType
from infy_gen_ai_fmwk.service.provider.chroma_db import TransformersVectorDB
from infy_gen_ai_fmwk.service.provider.llm_api import LLMAPI

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


class RetrieverProvider:
    def __init__(self):
        self.api_obj = LLMAPI()

    def retrieve_context(self, query, type, config_data):
        query_embed = self.api_obj._get_embedding(query, type, config_data)
        collection_name = ''
        if type == ModelType.OPENAI:
            collection_name = 'openai_embedding_collection'
        elif type == ModelType.OPENSOURCE:
            collection_name = 'opensource_embedding_collection'
        logger.info("Collection Name: %s", collection_name)
        vector_db = TransformersVectorDB(config_data)
        sorted_docs = vector_db.query_collection(
            query_embed, collection_name)
        context_length = 0
        closest_matches = []
        context = ''
        if len(sorted_docs['documents']) != 0:
            for i, document in enumerate(sorted_docs['documents'][0]):
                if context_length > config_data.context_length:
                    break
                context += f' {document}'
                context_length += len(document.split())
                distance = sorted_docs['distances'][0][i]
                closest_matches.append(
                    {'document': document, 'distance': distance})
                closest_matches = sorted(
                    closest_matches, key=lambda x: x['distance'])[:config_data.top_k_matches]
        logger.info("query: %s", query)
        logger.info("%d Closest Matches found", len(closest_matches))
        closest_matches_t = []
        if closest_matches != []:
            logger.debug("Closest Matches: %s", json.dumps(
                closest_matches, indent=2))
            closest_matches_t = [
                match for match in closest_matches if match['distance'] <= config_data.threshold_distance]
            logger.info("%d Closest Matches found within the threshold distance %.2f",
                        len(closest_matches_t), config_data.threshold_distance)
            logger.debug("Closest Matches found within the threshold distance: %s", json.dumps(
                closest_matches_t, indent=2))
        return closest_matches_t

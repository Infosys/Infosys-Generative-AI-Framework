# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for searching knowledge graph builder"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.data.request_data import SearchKnowledgeGraphBuilderRequestData
from infy_gen_ai_fmwk.data.response_data import SearchKnowledgeGraphBuilderResponseData
from infy_gen_ai_fmwk.service.provider.knowledge_graph_builder_provider import KnowledgeGraph

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

class SearchKnowledgeGraphBuilderService:
    def __init__(self, config_data):
        self.config_data = config_data
        logger.info('SearchKnowledgeGraphBuilderService initialized')

    def search_knowledge_graph(self, request_data: SearchKnowledgeGraphBuilderRequestData) -> SearchKnowledgeGraphBuilderResponseData:
        response_data = SearchKnowledgeGraphBuilderResponseData()
        try:
            if not self.config_data.root_path:
                logger.error(
                    FileConstants.ERROR_MSG_KG_SEARCH_OUTPUT_ROOT_PATH_EMPTY)
                raise ValueError(
                    FileConstants.ERROR_MSG_KG_SEARCH_OUTPUT_ROOT_PATH_EMPTY)
            kg_obj = KnowledgeGraph(request_data)
            graph_image_file_path, message_text = kg_obj.search_through_query(
                request_data.query, self.config_data)
            if message_text == "None":
                logger.info('Search Knowledge graph built successfully')
            else:
                logger.info('Search Knowledge graph not built successfully')
            response_data.graph_output_image_file_path = graph_image_file_path
            response_data.message_text = message_text
            return response_data
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise Exception(str(e))

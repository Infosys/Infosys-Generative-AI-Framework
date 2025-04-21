# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class to generate knowledge graph"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import KnowledgeGraphBuilderRequestData
from infy_gen_ai_fmwk.data.response_data import KnowledgeGraphBuilderResponseData
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

class KnowledgeGraphBuilderService:
    def __init__(self, config_data: OpenAiLlmConfigData):
        self.config_data = config_data
        logger.info('KnowledgeGraphBuilderService initialized')

    def generate_knowledge_graph(self, request_data: KnowledgeGraphBuilderRequestData) -> KnowledgeGraphBuilderResponseData:
        response_data = KnowledgeGraphBuilderResponseData()
        if not request_data.mode:
            logger.error(FileConstants.get_invalid_mode_type_error_msg())
            raise ValueError(FileConstants.get_invalid_mode_type_error_msg())
        try:
            if not self.config_data.root_path:
                logger.error(FileConstants.ERROR_MSG_KG_OUTPUT_ROOT_PATH_EMPTY)
                raise ValueError(
                    FileConstants.ERROR_MSG_KG_OUTPUT_ROOT_PATH_EMPTY)
            kg_obj = KnowledgeGraph(request_data)
            graph_image_file_path, graph_output_json_file_path, message_text = kg_obj.create_from_prompt(
                request_data.query, request_data.mode, self.config_data)
            if message_text == "None":
                logger.info('Search Knowledge graph built successfully')
            else:
                logger.info('Search Knowledge graph not built successfully')
            response_data.graph_output_image_file_path = graph_image_file_path
            response_data.graph_output_json_file_path = graph_output_json_file_path
            response_data.message_text = message_text
            return response_data
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise Exception(str(e))

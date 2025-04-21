# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import ast
import json
import os
import logging
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from infy_gen_ai_fmwk.data.request_data import ModeType
from infy_gen_ai_fmwk.service.provider.openai import OpenAIAPI
from logging.handlers import TimedRotatingFileHandler

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


class KnowledgeGraph:
    def __init__(self, request_data):
        self.graph_json_file_path = request_data.graph_json_file_path
        self.graph = nx.DiGraph()
        self.api = OpenAIAPI()

    def __load_json_file(self, config_data):
        if not self.graph_json_file_path:
            output_path = config_data.root_path
            os.makedirs(output_path, exist_ok=True)
            files = [f for f in os.listdir(
                config_data.root_path) if f.endswith(".json")]
            if files:
                latest_file = max(files, key=lambda x: os.path.getctime(
                    os.path.join(config_data.root_path, x)))
                logger.info(f"Latest file: {latest_file}")
                self.graph_json_file_path = os.path.join(
                    config_data.root_path, latest_file)
            else:
                return ""
        text_data = ""
        with open(self.graph_json_file_path, 'r') as file:
            json_data = json.load(file)
            for item in json_data:
                subject = item.get("subject", "")
                predicate = item.get("predicate", "")
                obj = item.get("object", "")
                text_data += f"('{subject}', '{predicate}', '{obj}'), "
        return text_data

    def __save_to_json_file(self, updated_graph_str, config_data):
        updated_graph_str = f"{updated_graph_str}"
        updated_graph = eval(updated_graph_str)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_json_file_path = os.path.join(
            config_data.root_path, f"{config_data.output_file_prefix}_{timestamp}.json")
        processed_items = set()
        json_data = []
        for item in updated_graph:
            logger.debug(f"Processing item: {item}")
            if isinstance(item, tuple) and len(item) == 3:
                if item not in processed_items:
                    subject, predicate, obj = item
                    json_data.append(
                        {"subject": subject, "predicate": predicate, "object": obj})
                    processed_items.add(item)
                else:
                    logger.info(f"Duplicate item found: {item}")
            else:
                logger.error(f"Invalid tuple length or type: {item}")
                raise ValueError(f"Invalid tuple length or type: {item}")
        with open(graph_json_file_path, 'w') as file:
            json.dump(json_data, file, indent=4)
        return graph_json_file_path

    def __prompt_generator(self, query, mode, config_data):
        if mode == ModeType.FILTER or mode == ModeType.filter:
            filter_prompt = config_data.openai_llm_config.prompt_template
            prompt = filter_prompt
            prompt += 'Prompt:' + query + '\n\n'
            prompt += 'Existing Graph: ' + \
                self.__load_json_file(config_data) + '\n\n'
            prompt += f'Filtered Graph:'
        if mode == ModeType.UPDATE or mode == ModeType.update:
            update_prompt = config_data.openai_llm_config.prompt_template
            prompt = update_prompt
            prompt += 'Prompt:' + query + '\n\n'
            prompt += 'Existing Graph: ' + \
                self.__load_json_file(config_data) + '\n\n'
            prompt += f'Updated Graph:'
        return prompt

    def __graph_generator(self, graph, config_data):
        knowledge_graph = ast.literal_eval(graph)
        self.graph = nx.DiGraph()
        for triplet in knowledge_graph:
            self.graph.add_edge(
                triplet[0].lower(), triplet[2].lower(), label=triplet[1].lower())
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        nx.draw(self.graph, pos, with_labels=True)
        labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_size=10, bbox=dict(
            facecolor='white', edgecolor='black', alpha=0.5))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config_data.root_path
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(
            config_data.root_path, f"{config_data.output_file_prefix}_{timestamp}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        graph_image_file_path = output_path
        plt.clf()
        return graph_image_file_path

    def create_from_prompt(self, query, mode, config_data):
        if mode == ModeType.UPDATE or mode == ModeType.update:
            logger.info(f"Query: {query}")
            prompt = self.__prompt_generator(query, mode, config_data)
            updated_graph = self.api.get_chat_completion(
                temperature=config_data.openai_llm_config.temperature, engine=config_data.openai_llm_config.completion_model, prompt=prompt)[0]
            logger.info(f"Updated graph:\n\n{updated_graph}\n\n")
            graph_image_file_path = self.__graph_generator(
                updated_graph, config_data)
            graph_output_json_file_path = self.__save_to_json_file(
                updated_graph, config_data)
            logger.info("Graph image saved to file: " + graph_image_file_path)
            logger.info("Json saved to file: " + graph_output_json_file_path)
            return [graph_image_file_path, graph_output_json_file_path, None]
        if mode == ModeType.FILTER or mode == ModeType.filter:
            logger.info(f"Query: {query}")
            prompt = self.__prompt_generator(
                query, mode, config_data)
            filtered_graph = self.api.get_chat_completion(
                temperature=config_data.openai_llm_config.temperature, engine=config_data.openai_llm_config.completion_model, prompt=prompt)[0]
            logger.info(f"filtered graph:\n\n{filtered_graph}\n\n")
            if filtered_graph == "[]":
                logger.info(f"No data found for the given query: {query}")
                message_text = "No data found for the given query: " + query
                return [None, None, message_text]
            else:
                graph_image_file_path = self.__graph_generator(
                    filtered_graph, config_data)
                logger.info("Graph image saved to file: " +
                            graph_image_file_path)
                return [graph_image_file_path, None, None]

    def __prompt_generator_recommendation(self, query, config_data):
        prompt_template = config_data.openai_llm_config.prompt_template
        prompt = str(self.__load_json_file(config_data))
        prompt += prompt_template.format(query=query)
        return prompt

    def search_through_query(self, query, config_data):
        prompt = self.__prompt_generator_recommendation(query, config_data)
        sub_graph = self.api.get_chat_completion(
            temperature=config_data.openai_llm_config.temperature, engine=config_data.openai_llm_config.completion_model, prompt=prompt)[0]
        if sub_graph == "[]":
            logger.info(f"No data found for the given query: {query}")
            message_text = "No data found for the given query: " + query
            return [None, message_text]
        else:
            graph_image_file_path = self.__graph_generator(
                sub_graph, config_data)
            logger.info("Graph image saved to file: " + graph_image_file_path)
            return [graph_image_file_path, None]

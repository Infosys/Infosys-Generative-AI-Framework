# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from infy_gen_ai_fmwk.data.request_data import ModelType
from infy_gen_ai_fmwk.service.provider.open_source import OpenSourceAPI
from infy_gen_ai_fmwk.service.provider.openai import OpenAIAPI

class LLMAPI:
    def __init__(self):
        self.openai_api = OpenAIAPI()
        self.opensource_api = OpenSourceAPI()

    def _get_completion(self, type, **kwargs):
        if type == ModelType.OPENAI:
            answers = self.openai_api.get_chat_completion(**kwargs)
        elif type == ModelType.OPENSOURCE:
            answers = self.opensource_api.get_chat_completion(**kwargs)
        return answers

    def _get_embedding(self, query: str, type, config_data):
        if type == ModelType.OPENAI:
            return self.openai_api.get_embedding(query, engine=config_data.embedding_config.model_name)
        elif type == ModelType.OPENSOURCE:
            return self.opensource_api.get_embedding(query, engine=config_data.embedding_config.model_path)

# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the OpenAI API provider class to interact with OpenAI API."""
import os
import openai
from dotenv import load_dotenv
load_dotenv()


class OpenAIAPI:
    """ OpenAI API provider class to interact with OpenAI API. """
    def __init__(self):
        openai.api_key = os.getenv("AZURE_OPENAI_KEY_EAST_US")
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
        openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    def get_chat_completion(self, prompt, temperature, engine, completions=1, stop=None):
        """Method to get chat completion from OpenAI API."""
        answers = []
        if completions > 1:
            if temperature == 0.0:
                temperature = 0.7

        response = openai.ChatCompletion.create(
            engine=engine,
            messages=[
                {"role": "system", "content": prompt},

            ]
        )
        answers.append(response.choices[0].message['content'])
        return answers

    def get_embedding(self, text, engine):
        """Method to get embedding from OpenAI API."""

        result = openai.Embedding.create(
            engine=engine,
            input=text
        )

        return result["data"][0]["embedding"]

# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData, SearchKnowledgeGraphBuilderConfigData
from infy_gen_ai_fmwk.data.request_data import SearchKnowledgeGraphBuilderRequestData
from infy_gen_ai_fmwk.service.search_knowledge_graph_builder_service import SearchKnowledgeGraphBuilderService

def test_search_knowledge_graph():
    request_data = SearchKnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query="",
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
        graph_json_file_path=""
    )
    config_data = SearchKnowledgeGraphBuilderConfigData(
        # root_path should be provided by the user
        root_path="C:/test",
        # output_file_prefix should be provided by the user
        output_file_prefix="kg_s",
        openai_llm_config=OpenAiLlmConfigData(
        # Prompt template needs to be changed according to the requirement.
        # The below is for reference prompt
        prompt_template="""
# From the above Knowledge graph for the below mentioned query,
# query: {query}\n
# Include details about the query in the form of tuples (subject, predicate, object):.
# If the query is not relevant to the provided information, the output will be []
""",    
     completion_model="gpt-4",
     open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
     open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
     open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    KnowledgeGraphBuilder = SearchKnowledgeGraphBuilderService(config_data)
    result = KnowledgeGraphBuilder.search_knowledge_graph(request_data)
    assert result is not None and result.graph_output_image_file_path is not None


def test_search_knowledge_graph_empty_file():
    request_data = SearchKnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query=""
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
    )
    config_data = SearchKnowledgeGraphBuilderConfigData(
        # root_path should be provided by the user
        root_path="C:/test",
        # output_file_prefix should be provided by the user
        output_file_prefix="kg_search",
        openai_llm_config=OpenAiLlmConfigData(
        # Prompt template needs to be changed according to the requirement
        # The below is for reference prompt    
        prompt_template="""
# From the above Knowledge graph for the below mentioned query,
# query: {query}\n
# Include details about the query in the form of tuples (subject, predicate, object):.
# If the query is not relevant to the provided information, the output will be []
""",
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    KnowledgeGraphBuilder = SearchKnowledgeGraphBuilderService(config_data)
    result = KnowledgeGraphBuilder.search_knowledge_graph(request_data)
    assert result is not None and result.graph_output_image_file_path is not None


def test_search_knowledge_graph_irrelevant_query():
    request_data = SearchKnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query=""
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
    )
    config_data = SearchKnowledgeGraphBuilderConfigData(
        # root_path should be provided by the user
        root_path="C:/test",
        # output_file_prefix should be provided by the user
        output_file_prefix="kg_search",
        openai_llm_config=OpenAiLlmConfigData(
        # Prompt template needs to be changed according to the requirement
        # The below is for reference prompt
        prompt_template="""
# From the above Knowledge graph for the below mentioned query,
# query: {query}\n
# Include details about the query in the form of tuples (subject, predicate, object):.
# If the query is not relevant to the provided information, the output will be []
""",
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    KnowledgeGraphBuilder = SearchKnowledgeGraphBuilderService(config_data)
    result = KnowledgeGraphBuilder.search_knowledge_graph(request_data)
    assert result is not None and result.message_text is not None


def test_search_knowledge_graph_empty_root_path():
    request_data = SearchKnowledgeGraphBuilderRequestData(
        # Query should be provided by the users
        query=""
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
    )
    config_data = SearchKnowledgeGraphBuilderConfigData(
        # output_file_prefix should be provided by the user
        output_file_prefix="kg_search",
        openai_llm_config=OpenAiLlmConfigData(
        # Prompt template needs to be changed according to the requirement
        # The below is for reference prompt
        prompt_template="""
# From the above Knowledge graph for the below mentioned query,
# query: {query}\n
# Include details about the query in the form of tuples (subject, predicate, object):.
# If the query is not relevant to the provided information, the output will be []
""",
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    with pytest.raises(ValueError):
        KnowledgeGraphBuilder = SearchKnowledgeGraphBuilderService(config_data)
        KnowledgeGraphBuilder.search_knowledge_graph(request_data)

# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.data.config_data import KnowledgeGraphBuilderConfigData, OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import KnowledgeGraphBuilderRequestData
from infy_gen_ai_fmwk.service.knowledge_graph_builder_service import KnowledgeGraphBuilderService


def test_generate_knowledge_graph_new():
    request_data = KnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query="",
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
        graph_json_file_path="",
        # Mode should be provided by the user
        mode="update"
    )
    config_data = KnowledgeGraphBuilderConfigData(
        # Root path should be provided by the user
        root_path="C:/test",
        # Output file prefix should be provided by the user
        output_file_prefix="kg_new",
        openai_llm_config=OpenAiLlmConfigData(
            prompt_template="""
# Given the current state of a graph and a prompt,
# Update the graph with relevant information.
# The graph is represented as a tuple of (subject, predicate, object).
# New relationships can be created by extracting predicates from the prompt.
# New relationships should be created by linking to existing nodes if relevant if not create new nodes.
# Example:
# Graph: [('COVID', 'Symptom', 'sweating'),('Cancer', 'Treatment', 'Chemotherapy')]
# Prompt : sulfur-drugs are chemicals used in chemotherapy
# Updated Graph: [('COVID', 'Symptom', 'sweating'),('Cancer', 'Treatment', 'Chemotherapy'),
#('Chemotherapy', 'chemicals', 'sulfur-drugs')]
# Example ends here
Graph:\n
""",
            completion_model="gpt-4",
            open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
            open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
            open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
            temperature=0.7
        )
    )
    KnowledgeGraphBuilder = KnowledgeGraphBuilderService(config_data)
    result = KnowledgeGraphBuilder.generate_knowledge_graph(request_data)
    assert result is not None and result.graph_output_image_file_path is not None and result.graph_output_json_file_path is not None


def test_generate_knowledge_graph_update():
    request_data = KnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query="",
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
        # Mode should be provided by the user
        mode="update"
    )
    config_data = KnowledgeGraphBuilderConfigData(
        # Root path should be provided by the user
        root_path="C:/test",
        # Output file prefix should be provided by the user
        output_file_prefix="kg_update",
        openai_llm_config=OpenAiLlmConfigData(
            prompt_template="""
# Given the current state of a graph and a prompt,
# Update the graph with relevant information.
# The graph is represented as a tuple of (subject, predicate, object).
# New relationships can be created by extracting predicates from the prompt.
# New relationships should be created by linking to existing nodes if relevant if not create new nodes.
# Example:
# Graph: [('COVID', 'Symptom', 'sweating'),('Cancer', 'Treatment', 'Chemotherapy')]
# Prompt : sulfur-drugs are chemicals used in chemotherapy
# Updated Graph: [('COVID', 'Symptom', 'sweating'),('Cancer', 'Treatment', 'Chemotherapy'),
# ('Chemotherapy', 'chemicals', 'sulfur-drugs')]
# Example ends here
Graph:\n""",
            completion_model="gpt-4",
            open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
            open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
            open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    KnowledgeGraphBuilder = KnowledgeGraphBuilderService(config_data)
    result = KnowledgeGraphBuilder.generate_knowledge_graph(request_data)
    assert result is not None and result.graph_output_image_file_path is not None and result.graph_output_json_file_path is not None


def test_generate_knowledge_graph_empty_json_path():
    request_data = KnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query="",
        # Mode should be provided by the user
        mode="update"
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
    )
    config_data = KnowledgeGraphBuilderConfigData(
        # Root path should be provided by the users
        root_path="C:/test",
        # Output file prefix should be provided by the user
        output_file_prefix="kg_update",
        openai_llm_config=OpenAiLlmConfigData(
            prompt_template="""
# Given the current state of a graph and a prompt,
# Update the graph with relevant information.
# The graph is represented as a tuple of (subject, predicate, object).
# New relationships can be created by extracting predicates from the prompt.
# New relationships should be created by linking to existing nodes if relevant if not create new nodes.
# Example:
# Graph: [('COVID', 'Symptom', 'sweating'),('Cancer', 'Treatment', 'Chemotherapy')]
# Prompt : sulfur-drugs are chemicals used in chemotherapy
# Updated Graph: [('COVID', 'Symptom', 'sweating'),('Cancer', 'Treatment', 'Chemotherapy'),
# ('Chemotherapy', 'chemicals', 'sulfur-drugs')]
# Example ends here
Graph:\n
""",
            completion_model="gpt-4",
            open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
            open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
            open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    KnowledgeGraphBuilder = KnowledgeGraphBuilderService(config_data)
    result = KnowledgeGraphBuilder.generate_knowledge_graph(request_data)
    assert result is not None and result.graph_output_image_file_path is not None


def test_generate_knowledge_graph_filter():
    request_data = KnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query="",
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
        graph_json_file_path="",
        # Mode should be provided by the user
        mode="filter",
    )
    config_data = KnowledgeGraphBuilderConfigData(
        # Root path should be provided by the user
        root_path="C:/test",
        # Output file prefix should be provided by the user
        output_file_prefix="kg_filter",
        openai_llm_config=OpenAiLlmConfigData(
            prompt_template="""
# Given the current state of a graph and a prompt,
# Filter out a sub graph which contains the relevant information from the prompt.
# The graph is represented as a tuple of (subject, predicate, object).
# Extract all tuples which are connected to the subject and object as well.
# Example
# Graph: [('Cancer', 'treatment', 'chemotherapy'), ('Diabetes', 'treatment', 'walking'),
# ('Diabetes', 'treatment', 'insulin'), ('Cancer', 'symptom', 'pain'),
# ('Chemotherapy', 'chemicals', 'sulfur')]
# Prompt: sulfur chemicals are used in chemotherapy
# Filtered Graph:  [('Cancer', 'treatment', 'chemotherapy'),
# ('Chemotherapy', 'chemicals', 'sulfur')]
Graph:\n
""",
            completion_model="gpt-4",
            open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
            open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
            open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    KnowledgeGraphBuilder = KnowledgeGraphBuilderService(config_data)
    result = KnowledgeGraphBuilder.generate_knowledge_graph(request_data)
    assert result is not None and result.graph_output_image_file_path is not None


def test_generate_knowledge_graph_irrelavant_query():
    request_data = KnowledgeGraphBuilderRequestData(
        # query should be provided by the user
        query="cat",
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
        graph_json_file_path="",
        # Mode should be provided by the user
        mode="filter",
    )
    config_data = KnowledgeGraphBuilderConfigData(
        # Root path should be provided by the user
        root_path="C:/test",
        # Output file prefix should be provided by the user
        output_file_prefix="knowledge_graph",
        openai_llm_config=OpenAiLlmConfigData(
            prompt_template="""
# Given the current state of a graph and a prompt,
# Filter out a sub graph which contains the relevant information from the prompt.
# The graph is represented as a tuple of (subject, predicate, object).
# Extract all tuples which are connected to the subject and object as well.
# Example
# Graph: [('Cancer', 'treatment', 'chemotherapy'), ('Diabetes', 'treatment', 'walking'),
# ('Diabetes', 'treatment', 'insulin'), ('Cancer', 'symptom', 'pain'),
# ('Chemotherapy', 'chemicals', 'sulfur')]
# Prompt: sulfur chemicals are used in chemotherapy
# Filtered Graph:  [('Cancer', 'treatment', 'chemotherapy'),
# ('Chemotherapy', 'chemicals', 'sulfur')]
Graph:\n
#If the query is not relevant to the graph, the output will be empty
""",
            completion_model="gpt-4",
            open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
            open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
            open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    KnowledgeGraphBuilder = KnowledgeGraphBuilderService(config_data)
    result = KnowledgeGraphBuilder.generate_knowledge_graph(request_data)
    assert result is not None and result.message_text is not None


def test_generate_knowledge_graph_with_no_mode():
    request_data = KnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query="",
    )
    config_data = KnowledgeGraphBuilderConfigData(
        # root path should be provided by the user
        root_path="C:/test",
        # output file prefix should be provided by the user
        output_file_prefix="kg",
        openai_llm_config=OpenAiLlmConfigData(
            prompt_template="""
# Given the current state of a graph and a prompt,
# Filter out a sub graph which contains the relevant information from the prompt.
# The graph is represented as a tuple of (subject, predicate, object).
# Extract all tuples which are connected to the subject and object as well.
# Example
# Graph: [('Cancer', 'treatment', 'chemotherapy'), ('Diabetes', 'treatment', 'walking'),
# ('Diabetes', 'treatment', 'insulin'), ('Cancer', 'symptom', 'pain'),
# ('Chemotherapy', 'chemicals', 'sulfur')]
# Prompt: sulfur chemicals are used in chemotherapy
# Filtered Graph:  [('Cancer', 'treatment', 'chemotherapy'),
# ('Chemotherapy', 'chemicals', 'sulfur')]
Graph:\n
""",
            completion_model="gpt-4",
            open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
            open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
            open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    with pytest.raises(ValueError):
        KnowledgeGraphBuilder = KnowledgeGraphBuilderService(config_data)
        KnowledgeGraphBuilder.generate_knowledge_graph(request_data)


def test_generate_knowledge_graph_with_empty_root_path():
    request_data = KnowledgeGraphBuilderRequestData(
        # Query should be provided by the user
        query="",
        # Mode should be provided by the user
        mode="filter",
        # User to provide graph_json_file_path if the search is required on a specific graph
        # If graph_json_file_path is not provided the search will happen on the latest json file from the root path
        graph_json_file_path=""
    )
    config_data = KnowledgeGraphBuilderConfigData(
        # output file prefix should be provided by the user
        output_file_prefix="kg",
        openai_llm_config=OpenAiLlmConfigData(
            prompt_template="""
# Given the current state of a graph and a prompt,
# Filter out a sub graph which contains the relevant information from the prompt.
# The graph is represented as a tuple of (subject, predicate, object).
# Extract all tuples which are connected to the subject and object as well.
# Example
# Graph: [('Cancer', 'treatment', 'chemotherapy'), ('Diabetes', 'treatment', 'walking'),
# ('Diabetes', 'treatment', 'insulin'), ('Cancer', 'symptom', 'pain'),
# ('Chemotherapy', 'chemicals', 'sulfur')]
# Prompt: sulfur chemicals are used in chemotherapy
# Filtered Graph:  [('Cancer', 'treatment', 'chemotherapy'),
# ('Chemotherapy', 'chemicals', 'sulfur')]
Graph:\n
""",
            completion_model="gpt-4",
            open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
            open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
            open_ai_version=os.environ['AZURE_OPENAI_API_VERSION']
        )
    )
    with pytest.raises(ValueError):
        KnowledgeGraphBuilder = KnowledgeGraphBuilderService(config_data)
        KnowledgeGraphBuilder.generate_knowledge_graph(request_data)

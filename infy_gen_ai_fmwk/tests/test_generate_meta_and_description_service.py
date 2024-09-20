# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.service.generate_meta_and_description_service import GenerateMetaAndDescriptionService
from infy_gen_ai_fmwk.data.config_data import OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import MetaAndDescriptionGeneratorRequestData

def test_generate_meta_and_description_service_valid_extension():
    # Input file path should be updated by the user
    test_file_path = ""
    request_data = MetaAndDescriptionGeneratorRequestData()
    request_data.file_path = test_file_path
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="""From the below mentioned Blog Content extract important keywords which can be further used to create a meta description for an attractive search result on Google India in 2020 from the given Keywords.
                        Blog Content:\n{}
                        Important Keywords : {}
                        From the below mentioned Blog Content and Important Keywords, Generate a Meta Description for an attractive search result on Google India in 2020. It must follow the Characteristics of a good Meta Descriptions.
                        Characteristics of a good Meta Descriptions are it is unique, it matches the content of the Blog, Show specifications when needed, focus on keyphrase, include call to action, uses active voice and make it actionable and it contains upto 155 characters only.
                        Blog Content:\n{} """
    )

    summary_service = GenerateMetaAndDescriptionService(config_data)
    result = summary_service.get_meta_description(request_data)
    assert result is not None and result.meta_descr_response is not None


def test_generate_meta_and_description_service_invalid_extension():
    # Input file path having invalid extension can be updated by the user
    test_file_path = "data/sample/input/blog.mp3"
    request_data = MetaAndDescriptionGeneratorRequestData()
    request_data.file_path = test_file_path
    config_data = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="""From the below mentioned Blog Content extract important keywords which can be further used to create a meta description for an attractive search result on Google India in 2020 from the given Keywords.
                        Blog Content: {}
                        Important Keywords : {}
                        From the below mentioned Blog Content and Important Keywords, Generate a Meta Description for an attractive search result on Google India in 2020. It must follow the Characteristics of a good Meta Descriptions.
                        Characteristics of a good Meta Descriptions are it is unique, it matches the content of the Blog, Show specifications when needed, focus on keyphrase, include call to action, uses active voice and make it actionable and it contains upto 155 characters only.
                        Blog Content: {}"""
                        )

    summary_service = GenerateMetaAndDescriptionService(config_data)
    with pytest.raises(ValueError):
        summary_service.get_meta_description(request_data)

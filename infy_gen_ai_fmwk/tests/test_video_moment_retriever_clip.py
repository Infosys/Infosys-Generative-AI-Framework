# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import pytest
from infy_gen_ai_fmwk.data.config_data import OpenSourceLlmConfigData, VideoMomentRetrieverConfigData
from infy_gen_ai_fmwk.data.request_data import VideoMomentRetrieverRequestData
from infy_gen_ai_fmwk.service.video_moment_retriever_clip_service import VideoMomentRetrieverClipService


def test_retrieve_video_moment():
    request_data = VideoMomentRetrieverRequestData(
        # Query should be updated by the user
        query="",
        # Display resultscount should be updated by the user
        display_results_count=1,
        # Video file path should be updated by the user
        video_file_path="",
        # Clip duration should be updated by the user
        clip_duration=5
    )
    config_data = VideoMomentRetrieverConfigData(
        # Output root path should be updated by the user
        output_root_path="C:/test/output",
        similarity_threshold=0.27
    )
    config_data.opensource_llm_config = OpenSourceLlmConfigData(
        # Steps to download a model can be referred from README.md file.The model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/clip-ViT-B-32"
    )
    video_moment_retriever = VideoMomentRetrieverClipService(config_data)
    result = video_moment_retriever.retrieve_video_moments(request_data)
    assert result is not None and result.output_file_paths is not None


def test_retrieve_video_moment_with_failed_query():
    request_data = VideoMomentRetrieverRequestData(
        # Query should be updated by the user
        query="Rally length is equal to 26",
        # Display results count should be updated by the user
        display_results_count=1,
        # Video file path should be updated by the user
        video_file_path="",
        # Clip duration should be updated by the user
        clip_duration=5
    )
    config_data = VideoMomentRetrieverConfigData(
        # Output root path should be updated by the user
        output_root_path="C:/test/output",
        similarity_threshold=0.27
    )
    config_data.opensource_llm_config = OpenSourceLlmConfigData(
        # Steps to download a model can be referred from README.md file.The model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/clip-ViT-B-32"
    )
    video_moment_retriever = VideoMomentRetrieverClipService(config_data)
    result = video_moment_retriever.retrieve_video_moments(request_data)
    # Expected output : The Video clip has one image with the query "Rally length is equal to 26"
    # Actual output:The Video clip has no image with the query "Rally length is equal to 26"
    assert result is not None and result.output_file_paths is not None


def test_retrieve_video_moment_with_rare_moment():
    request_data = VideoMomentRetrieverRequestData(
        # Query should be updated by the user
        query="",
        # Display results count should be updated by the user
        display_results_count=1,
        # Video file path should be updated by the user
        video_file_path="",
        # Clip duration should be updated by the user
        clip_duration=5
    )
    config_data = VideoMomentRetrieverConfigData(
        # Output root path should be updated by the user
        output_root_path="C:/test/output",
        similarity_threshold=0.27
    )
    config_data.opensource_llm_config = OpenSourceLlmConfigData(
        # Steps to download a model can be referred from README.md file.The model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/clip-ViT-B-32"
    )
    video_moment_retriever = VideoMomentRetrieverClipService(config_data)
    result = video_moment_retriever.retrieve_video_moments(request_data)
    assert result is not None and result.output_file_paths is not None


def test_retrieve_video_moment_invalid_query():
    request_data = VideoMomentRetrieverRequestData(
        # Query should be updated by the user
        query="",
        # Display results count should be updated by the user
        display_results_count=2,
        # Video file path should be updated by the user
        video_file_path="",
        # Clip duration should be updated by the user
        clip_duration=5
    )
    config_data = VideoMomentRetrieverConfigData(
        # Output root path should be updated by the user
        output_root_path="C:/test/output",
        similarity_threshold=0.27
    )
    config_data.opensource_llm_config = OpenSourceLlmConfigData(
        # Steps to download a model can be referred from README.md file.The model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/clip-ViT-B-32"
    )
    video_moment_retriever = VideoMomentRetrieverClipService(config_data)
    result = video_moment_retriever.retrieve_video_moments(request_data)
    assert result is not None and result.output_file_paths is not None


def test_retrieve_video_moment_with_invalid_type():
    request_data = VideoMomentRetrieverRequestData(
        # Query should be updated by the user
        query="audience clapping",
        # Display results count should be updated by the user
        display_results_count=5,
        # Video file path having invalid extension can be updated by the user
        video_file_path="data/sample/input/10_Times_Roger_Federer_Blew_Our_Minds_TOP_10_ATP.mp3",
        # Clip duration should be updated by the user
        clip_duration=5
    )
    config_data = VideoMomentRetrieverConfigData(
        # Output root path should be updated by the user
        output_root_path="C:/test/output",
        similarity_threshold=0.27
    )
    config_data.opensource_llm_config = OpenSourceLlmConfigData(
        # Steps to download a model can be referred from README.md file.The model_path below needs to be updated by the user accordingly.
        model_path="C:/MyProgramFiles/AI/models/clip-ViT-B-32"
    )
    with pytest.raises(ValueError):
        video_moment_retriever = VideoMomentRetrieverClipService(config_data)
        video_moment_retriever.retrieve_video_moments(request_data)

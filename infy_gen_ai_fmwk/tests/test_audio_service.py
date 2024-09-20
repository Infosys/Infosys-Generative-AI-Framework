# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import os
import pytest
from infy_gen_ai_fmwk.service.audio_to_text_service import AudioToTextService
from infy_gen_ai_fmwk.data.config_data import AudioToTextConfigData, OpenAiLlmConfigData
from infy_gen_ai_fmwk.data.request_data import AudioToTextRequestData


def test_convert_to_text_with_mom():
    # Input file path should be updated by the user
    test_file_path = ""
    open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate detailed minutes of meeting from the below conversation and list the action items: {audio_text}\nMOM:"
    )
    request_data = AudioToTextRequestData(
        file_path=test_file_path,
        is_mom_required=True
    )
    config_data = AudioToTextConfigData(
        #Steps to download a model can be referred from README.md file.The model_path below needs to be updated by the user accordingly.
        audio_converter_model_path="C:/MyProgramFiles/AI/models/whisper-base",
        compressed_audio_filepath="audio.flac",
        audio_chunck_length=30,
        audio_Target_sr=16000,
        audio_batch_size=3,
        open_ai_config=open_ai_config
    )

    audio_service = AudioToTextService(config_data)
    result = audio_service.convert_to_text(request_data)
    assert result is not None and result.transcription_response_txt is not None and result.mom_text is not None


def test_convert_to_text_without_mom():
    # Input file path should be updated by the user
    test_file_path = ""
    open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate detailed minutes of meeting from the below conversation and list the action items: {audio_text}\nMOM:"
    )
    request_data = AudioToTextRequestData(
        file_path=test_file_path,
        is_mom_required=False
    )
    config_data = AudioToTextConfigData(
        #Steps to download a model can be referred from README.md file.The model_path below needs to be updated by the user accordingly.
        audio_converter_model_path="C:/MyProgramFiles/AI/models/whisper-base",
        compressed_audio_filepath="audio.flac",
        audio_chunck_length=30,
        audio_Target_sr=16000,
        audio_batch_size=3,
        open_ai_config=open_ai_config
    )

    audio_service = AudioToTextService(config_data)
    result = audio_service.convert_to_text(request_data)
    assert result is not None and result.transcription_response_txt is not None and result.mom_text is None


def test_invalid_extension():
    # Input file path having invalid extension can be updated by the user
    test_file_path = "data/sample/input/MOMSampleRecording.txt"
    open_ai_config = OpenAiLlmConfigData(
        completion_model="gpt-4",
        open_ai_url=os.environ['AZURE_OPENAI_API_BASE'],
        open_ai_key=os.environ['AZURE_OPENAI_KEY_EAST_US'],
        open_ai_version=os.environ['AZURE_OPENAI_API_VERSION'],
        prompt_template="Generate detailed minutes of meeting from the below conversation and list the action items: {audio_text}\nMOM:"
    )
    request_data = AudioToTextRequestData(
        file_path=test_file_path,
        is_mom_required=False
    )
    config_data = AudioToTextConfigData(
        #Steps to download a model can be referred from README.md file.The model_path below needs to be updated by the user accordingly.
        audio_converter_model_path="C:/MyProgramFiles/AI/models/whisper-base",
        compressed_audio_filepath="audio.flac",
        audio_chunck_length=30,
        audio_Target_sr=16000,
        audio_batch_size=3,
        open_ai_config=open_ai_config
    )
    audio_service = AudioToTextService(config_data)
    with pytest.raises(ValueError):
        audio_service.convert_to_text(request_data)

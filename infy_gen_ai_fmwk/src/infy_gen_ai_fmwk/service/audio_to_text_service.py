# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for converting audio to text"""
import os
import io
import logging
from logging.handlers import TimedRotatingFileHandler
import soundfile as sf
from infy_gen_ai_fmwk.data.request_data import AudioToTextRequestData, AudioToTextResponseData
from infy_gen_ai_fmwk.data.config_data import AudioToTextConfigData
from infy_gen_ai_fmwk.service.provider.openai import OpenAIAPI
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.service.provider.audio_translator import AudioTranslator
from infy_gen_ai_fmwk.common.file_constants import FileConstants
if not os.path.exists('logs'):
    os.makedirs('logs')
logger = logging.getLogger('logger')
logger.setLevel(logging.ERROR)
handler = TimedRotatingFileHandler(
    'logs/gen_ai_fmwk.log', when='midnight', interval=1)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class AudioToTextService:
    """ Service class for converting audio to text """

    def __init__(self, config_data: AudioToTextConfigData):
        self.config_data = config_data
        self.audio_translator_obj = AudioTranslator(config_data)
        logger.info('AudioToTextService initialized')

    def convert_to_text(self, request_data: AudioToTextRequestData) -> AudioToTextResponseData:
        """ Method to convert audio to text """
        response_data = AudioToTextResponseData()
        if not Utils.check_extension(request_data.file_path, FileConstants.VALID_EXTENSIONS['audio_file']):
            logger.error(
                FileConstants.get_invalid_file_error_msg('audio_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('audio_file'))

        with open(request_data.file_path, 'rb') as f:
            audio_bytes = f.read()
        signal, sr = sf.read(io.BytesIO(audio_bytes))
        sf.write(self.config_data.compressed_audio_filepath, signal, sr)
        try:
            logger.info('converting audio to text')
            response_data.transcription_response_txt = self.audio_translator_obj.audio_to_text(
                self.config_data)
            logger.info('Audio to text conversion completed')
            if request_data.is_mom_required:
                api = OpenAIAPI()
                llm_config = self.config_data.open_ai_config
                prompt = llm_config.prompt_template.format(
                    audio_text=response_data.transcription_response_txt)
                answers = api.get_chat_completion(
                    prompt, llm_config.temperature, engine=llm_config.completion_model)
                response_data.mom_text = answers[0]
                logger.info('Generated MOM')
            return response_data
        except Exception as e:
            logger.error('Error in converting audio to text')
            raise Exception(str(e))

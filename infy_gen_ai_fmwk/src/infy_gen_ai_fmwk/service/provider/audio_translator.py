# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

""" This module is used to convert audio to text using the Hugging Face pipeline. """
import librosa as lr
from transformers import pipeline
from infy_gen_ai_fmwk.data.config_data import AudioToTextConfigData
class AudioTranslator:
    """ This class is used to convert audio to text using the Hugging Face pipeline. """

    def __init__(self, config_data: AudioToTextConfigData):
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=config_data.audio_converter_model_path,
            chunk_length_s=config_data.audio_chunck_length,

        )

    def audio_to_text(self, config_data: AudioToTextConfigData):
        """ This method is used to convert audio to text using the Hugging Face pipeline. """
        data, sample_rate = lr.load(config_data.compressed_audio_filepath)
        data = lr.resample(data, orig_sr=sample_rate,
                           target_sr=config_data.audio_Target_sr)

        prediction = self.pipeline(
            data.copy(), batch_size=config_data.audio_batch_size)["text"]

        return prediction

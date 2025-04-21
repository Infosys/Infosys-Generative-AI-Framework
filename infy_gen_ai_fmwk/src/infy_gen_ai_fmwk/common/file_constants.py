# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from infy_gen_ai_fmwk.data.request_data import ModeType, ModelType


class FileConstants:
    """ Class to define file constants """
    VALID_EXTENSIONS = {
        'source_file': ['.txt', '.java', '.py', '.c', '.c#', '.html', '.cpp', '.js', '.php', '.rb', '.go', '.swift', '.kt', '.rs'],
        'audio_file': ['.mp3', '.wav'],
        'pdf_file': ['.pdf'],
        'excel_file': ['.csv', '.xlsx'],
        'txt_file': ['.txt'],
        'doc_file': ['.docx'],
        'video_file': ['.mp4']
    }

    @staticmethod
    def get_invalid_file_error_msg(file_type):
        """ Method to get invalid file error message """
        valid_extensions = ', '.join(FileConstants.VALID_EXTENSIONS[file_type])
        return f'No valid file(s) to process. Valid extension(s) are: {valid_extensions}'

    PIPELINE_TASK_QA = 'question-answering'

    @staticmethod
    def get_invalid_model_type_error_msg():
        return f'Model type cannot be None or an empty string. Defaulting to {ModelType.OPENAI.value}'

    FPS = 30
    CODEC = "libx264"
    BATCH_SIZE = 256

    @staticmethod
    def get_invalid_mode_type_error_msg():
        return f'Invalid mode type. Valid mode types are: {ModeType.UPDATE.value}, {ModeType.FILTER.value},{ModeType.filter.value},{ModeType.update.value}'

    ERROR_MSG_KG_OUTPUT_ROOT_PATH_EMPTY = "Output root path should not be empty for knowledge graph"
    ERROR_MSG_KG_SEARCH_OUTPUT_ROOT_PATH_EMPTY = "Output root path should not be empty for searching knowledge graph"

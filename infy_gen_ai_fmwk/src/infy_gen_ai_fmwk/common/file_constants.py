# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

class FileConstants:
    """ Class to define file constants """
    VALID_EXTENSIONS = {
        'source_file': ['.txt', '.java', '.py', '.c', '.c#', '.html', '.cpp', '.js', '.php', '.rb', '.go', '.swift', '.kt', '.rs'],
        'audio_file': ['.mp3', '.wav'],
        'pdf_file': ['.pdf'],
        'excel_file': ['.csv', '.xlsx'],
        'txt_file': ['.txt'],
        'doc_file': ['.docx']
    }

    @staticmethod
    def get_invalid_file_error_msg(file_type):
        """ Method to get invalid file error message """
        valid_extensions = ', '.join(FileConstants.VALID_EXTENSIONS[file_type])
        return f'No valid file(s) to process. Valid extension(s) are: {valid_extensions}'

# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#


class Utils:
    """ Utility class for common methods """

    def readTxt(file_path):
        """ Method to read text from file """
        with open(file_path, 'r') as file:
            return file.read()

    def check_extension(filename, allowed_ext):
        """ Method to check if the file extension is allowed """
        if filename.lower().endswith(tuple(allowed_ext)):
            return True

        return False

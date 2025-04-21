# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

"""This file contains the service class for retrieving video"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from infy_gen_ai_fmwk.common.file_constants import FileConstants
from infy_gen_ai_fmwk.common.utils import Utils
from infy_gen_ai_fmwk.data.config_data import VideoMomentRetrieverConfigData
from infy_gen_ai_fmwk.data.request_data import VideoMomentRetrieverRequestData
from infy_gen_ai_fmwk.data.response_data import VideoMomentRetrieverResponseData
from infy_gen_ai_fmwk.service.provider.video_moment_retriever_clip_provider import VideoMomentRetrieverClipProvider

if not os.path.exists('logs'):
    os.makedirs('logs')
logger = logging.getLogger('logger')
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = TimedRotatingFileHandler(
        'logs/gen_ai_fmwk.log', when='midnight', interval=1)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class VideoMomentRetrieverClipService:
    def __init__(self, config_data: VideoMomentRetrieverConfigData):
        self.config_data = config_data
        logger.info('VideoMomentRetrieverClipService initialized')

    def retrieve_video_moments(self, request_data: VideoMomentRetrieverRequestData) -> VideoMomentRetrieverResponseData:
        response_data = VideoMomentRetrieverResponseData()
        if not Utils.check_extension(request_data.video_file_path, FileConstants.VALID_EXTENSIONS['video_file']):
            logger.error(
                FileConstants.get_invalid_file_error_msg('video_file'))
            raise ValueError(
                FileConstants.get_invalid_file_error_msg('video_file'))
        try:
            obj = VideoMomentRetrieverClipProvider(
                request_data, self.config_data)
            result = obj.text_search(
                request_data.query, request_data.display_results_count)
            logger.info('Retrieved video moments and images')
            response_data.output_file_paths = result
            return response_data
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            raise Exception(str(e))

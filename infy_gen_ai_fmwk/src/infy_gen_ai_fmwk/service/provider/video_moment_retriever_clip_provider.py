# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import math
import os
import cv2
import pandas as pd
import torch
import logging
from logging.handlers import TimedRotatingFileHandler
from humanfriendly import format_timespan
from moviepy.editor import VideoFileClip
from PIL import Image
from sentence_transformers import SentenceTransformer
from infy_gen_ai_fmwk.common.file_constants import FileConstants

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


class VideoMomentRetrieverClipProvider():
    def __init__(self, request_data, config_data):
        self.device = 'cpu'
        self.video_file_path = request_data.video_file_path
        self.output_dir = config_data.output_root_path
        self.video_features = os.path.join(
            self.output_dir, "video_features.pkl")
        self.video_details = os.path.join(
            self.output_dir, "video_details.pkl")
        self.similarity_threshold = config_data.similarity_threshold
        self.clip_duration = request_data.clip_duration
        self.model = SentenceTransformer(
            config_data.opensource_llm_config.model_path, device=self.device)
        self.video_features_dic = {}
        self.__create_video_features()
        self.__load_video_features()

    def __create_video_features(self):
        video_frames = []
        video_features = None
        video_details = pd.DataFrame()
        video_path = os.path.join(self.video_file_path)
        file = os.path.basename(video_path)
        frames, fps, duration = self.__extract_frames(video_path)
        features = self.__encode_frames(frames)
        # Updates the frames and features
        video_frames.append(frames)
        if torch.is_tensor(video_features):
            video_features = torch.cat((video_features, features), dim=0)
        else:
            video_features = features
        # Update the fps and duration
        video_details_row = pd.DataFrame(
            [{'video_name': file, 'fps': fps, 'duration': duration}])
        video_details = pd.concat(
            [video_details, video_details_row], ignore_index=True)
        output_dir_features = os.path.dirname(self.video_features)
        if not os.path.exists(output_dir_features):
            os.makedirs(output_dir_features)
        output_dir_details = os.path.dirname(self.video_details)
        if not os.path.exists(output_dir_details):
            os.makedirs(output_dir_details)
        df = pd.DataFrame([{'video_features': video_features.numpy()}])
        df.to_pickle(self.video_features)
        video_details.to_pickle(self.video_details)

    def __load_video_features(self):
        df = pd.read_pickle(self.video_features)
        self.video_features_dict = torch.from_numpy(
            df.iloc[0]['video_features'])
        self.video_details = pd.read_pickle(self.video_details)

    def __extract_frames(self, video):
        frames = []
        capture = cv2.VideoCapture(video)
        fps = capture.get(cv2.CAP_PROP_FPS)
        num_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        current_frame = 0
        while capture.isOpened():
            ret, frame = capture.read()
            if ret == True:
                frames.append(Image.fromarray(frame[:, :, ::-1]))
            else:
                break
            current_frame += 30
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        duration = round(num_frames / round(fps))
        return frames, fps, duration

    def __encode_frames(self, video_frames):
        batches = math.ceil(len(video_frames) / FileConstants.BATCH_SIZE)
        video_features = torch.empty(
            [0, 512], dtype=torch.float16).to(self.device)
        for i in range(batches):
            batch_frames = video_frames[i *
                                        FileConstants.BATCH_SIZE: (i+1)*FileConstants.BATCH_SIZE]
            batch_preprocessed = [self.model.encode(
                frame) for frame in batch_frames]
            batch_features = torch.tensor(batch_preprocessed).to(self.device)
            video_features = torch.cat((video_features, batch_features))
        return video_features

    def __display_results(self, best_photo_idx, input_query):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        video_frames = []
        no_of_frames = []
        frame_in_video = 0
        video_id = 0
        video_path = os.path.join(self.video_file_path)
        file = os.path.basename(video_path)
        frames, _, _ = self.__extract_frames(video_path)
        video_frames.extend(frames)
        no_of_frames.append(len(frames))
        for index, frame_id in enumerate(best_photo_idx):
            frame_id = frame_id.item()
            result = video_frames[frame_id]
            frame_number = frame_id
            frame_in_video = frame_number
            seconds = round(frame_in_video * 30 /
                            round(self.video_details.iloc[video_id]['fps']))
            input_video_name = self.video_details.iloc[video_id]['video_name']
            logger.info(
                f"Selected video: {file}, Frame in video: {frame_in_video}, Seconds: {seconds}, Input video name: {input_video_name}")
            logger.info(
                f"Processed frame {frame_in_video} from video {file}")
            input_video_base_name = os.path.splitext(input_video_name)[0]
            input_video = VideoFileClip(os.path.join(
                self.video_file_path))
            start_time = seconds
            end_time = seconds + self.clip_duration
            clip = input_video.subclip(start_time, end_time)
            time = format_timespan(seconds)
            output_dir_path = os.path.join(
                self.output_dir, input_video_base_name)
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            images_output_dir = os.path.join(output_dir_path, "images")
            videos_output_dir = os.path.join(output_dir_path, "videos")
            if not os.path.exists(images_output_dir):
                os.makedirs(images_output_dir)
            if not os.path.exists(videos_output_dir):
                os.makedirs(videos_output_dir)
            clip.write_videofile(os.path.join(
                videos_output_dir, f'output_{index + 1}.mp4'), FileConstants.FPS, FileConstants.CODEC, audio=False, rewrite_audio=False, logger=None)
            result.save(os.path.join(images_output_dir,
                                     input_query + "_" + str(time) + ".jpeg"))
        return [images_output_dir, videos_output_dir]

    def text_search(self, search_query, display_results_count):
        with torch.no_grad():
            text_features = torch.tensor(
                self.model.encode(search_query)).to(self.device)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logger.debug(f"Text Features: {text_features}")
            video_features_norm = self.video_features_dict / \
                self.video_features_dict.norm(dim=-1, keepdim=True)
            logger.debug(f"Normalized Video Features: {video_features_norm}")
            similarities = (video_features_norm @ text_features.T).squeeze()
            logger.debug(f"Similarities: {similarities}")
            logger.debug(f"Similarities: {similarities}")
            filtered_indices = (similarities >= self.similarity_threshold).nonzero(
                as_tuple=True)[0]
            filtered_similarities = similarities[filtered_indices]
            logger.debug(f"Filtered similarities: {filtered_similarities}")
            if len(filtered_similarities) == 0:
                logger.info("No relevant frames found for the query.")
                return [None, None]
            topk_similarities, best_photo_idx = torch.topk(filtered_similarities, min(
                display_results_count, len(filtered_similarities)), dim=0)
            logger.debug(f"Similarity threshold: {self.similarity_threshold}")
            for i, similarity in enumerate(topk_similarities):
                logger.debug(f"Top {i+1} similarity: {similarity.item()}")
            logger.info("Relevant frames found for the query.")
            images_output_dir, videos_output_dir = self.__display_results(
                filtered_indices[best_photo_idx], search_query)
            return [images_output_dir, videos_output_dir]

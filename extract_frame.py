import os
from os.path import exists, join, expanduser
import cv2
import wget

def video_download(videos_urls, videos_path):
    if not exists(videos_path):
        os.mkdir(videos_path)

    for i in range(len(videos_urls)):
        data = 'https://emotionstudy.s3.ap-south-1.amazonaws.com/final3/' + videos_urls.video_id[i] + '_1min.mp4'
        wget.download(data, videos_path)

    print('videos downloaded')

def extract_frames(frame_folder, frame_rate, videos_path):
    if not exists(frame_folder):
        os.mkdir(frame_folder)

    for video_name in os.listdir(videos_path):
        if video_name.endswith('.mp4'):
            #the frames for each video will be saved in this folder
            video_frame_folder = os.path.join(frame_folder, video_name[:-9])

            if not exists(video_frame_folder):
                os.mkdir(video_frame_folder)

            video_capture = cv2.VideoCapture(os.path.join(videos_path, video_name))
            current_frame = 0 #to match existing code, starting frame is 0
            sec = 0

            #read each frame and save it
            while(True):
                video_capture.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
                has_frames, img = video_capture.read()

                if has_frames:
                    frame_name = video_name[:-9] + '_' + str(current_frame) + '.jpg' # os.path.splitext(video_name)[0]
                    cv2.imwrite(os.path.join(video_frame_folder, frame_name), img)
                    current_frame += 1
                    sec += frame_rate
                    sec = round(sec, 2)
                else:
                    break

    print('frames for all videos have been exported')

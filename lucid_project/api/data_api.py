
import numpy as np
import requests
import time
import os
import json

class DataFile:
	pass

class DataFrame:
	pass

BASE_PATH = "http://starserver.thelangton.org.uk/lucid_dashboard/data_api/"

def get_data_files(run = None):
        data_path = os.path.join(os.path.dirname(__file__), "data/files.json")
        with open(data_path) as json_data:
                files = json.load(json_data)
        if run:
                filtered_files = []
                for data_file in files:
                        if data_file['run'] == run:
                                filtered_files.append(data_file)
                return filtered_files
        return files

def get_runs():
        data_path = os.path.join(os.path.dirname(__file__), "data/files.json")
        with open(data_path) as json_data:
            files = json.load(json_data)
        runs = []
        for data_file in files:
                if not data_file['run'] in runs:
                        runs.append(data_file['run'])
        return runs


def get_frames(file_id):
        response = requests.get(BASE_PATH + "get/frames/" + str(int(file_id)))
        time.sleep(1) 
        if not response.ok:
                raise Exception("That data file could not be found")
        frames = response.json()
        updated_frames = []

        for frame in frames:
                frame_obj = DataFrame()
                frame_obj.__dict__ = frame
                frame = frame_obj
                new_channels = []
                for channel_id in range(5):
                        channel = np.zeros((256, 256))
                        chip = "c"+str(channel_id)
                        if chip in frame.channels.keys():
                                for line in frame.channels[chip].split("\n")[:-1]: # Last line is blank
                                        vals = line.split("\t")
                                        x = int(float(vals[0].strip()))
                                        y = int(float(vals[1].strip()))
                                        c = int(float(vals[2].strip()))
                                        channel[x][y] = c
                                new_channels.append(channel)
                        else:
                                new_channels.append(None)
                frame.channels = new_channels
                updated_frames.append(frame)
        return updated_frames

#!/usr/bin/env python3
# -*- coding: utf_8 -*-
import pip

# Install packages
# pip.main(["install", "-U", "-q", "-r", "requirements.txt"])

import csv
import os
import shutil
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Load global variable from .env file
load_dotenv()

if __name__ == '__main__':
    # Load a detect model
    if os.getenv("TASK") == "detect":
        # model = YOLO("yolov8n.pt")
        # model = YOLO("runs/detect/train/weights/best.pt")
        # model = YOLO("runs/detect/train2/weights/best.pt")
        model = YOLO("runs/detect/train3/weights/best.pt")
    elif os.getenv("TASK") == "classify":
        model = YOLO("yolov8n-cls.pt")
    elif os.getenv("TASK") == "segment":
        model = YOLO("yolov8n-seg.pt")
    else:
        raise ValueError("Please set correct TASK in .env file")

    if os.getenv("MODE") == "train":
        model.train(data=os.getenv("DATA"), epochs=int(os.getenv("EPOCHS")))
    elif os.getenv("MODE") == "predict":
        # Use the model
        # Test image: https://ultralytics.com/images/bus.jpg
        # results = model.predict(show=True, source="https://ultralytics.com/images/bus.jpg",
        #                         save=True, save_crop=True, save_txt=True)
        results = model.predict(show=True, source="inputs/16.png",
                                save=True, save_crop=True, save_txt=True, hide_labels=True)
        # results = model.predict(show=True, source="0")  # source="0" for webcam
        exit(0)

        config = Cfg.load_config_from_name("vgg_transformer") # "vgg_seq2seq"
        # config = Cfg.load_config_from_name("vgg_seq2seq") # "vgg_transformer"
        config["device"] = "cpu"

        detector = Predictor(config)

        # Write the data to a CSV file
        src_dir = "./src"
        predict_folder = "./runs/detect/predict"
        detect_folder = "./runs/detect/predict/crops/drugname"
        folder_path = os.path.abspath(predict_folder)
        with open(os.getenv("CSV_FILE"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Drugname"])
            for image in sorted(os.listdir(src_dir), key=lambda x: os.path.getmtime(os.path.join(src_dir, x))):
                results = model.predict(source=src_dir+"/"+image, save=True, save_crop=True, hide_labels=True)
                lst = []
                for crop in sorted(os.listdir(detect_folder)):
                    drugname = detector.predict(Image.open(detect_folder + "/" + crop))
                    drugname = drugname[drugname.find(")") + 1:].strip()
                    if drugname:
                        lst.append([image, drugname])
                writer.writerows(lst)
                os.chmod(folder_path, 0o777)
                shutil.rmtree(folder_path)
    else:
        raise ValueError("Please set correct MODE in .env file")
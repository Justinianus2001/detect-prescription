#!/usr/bin/env python3
# -*- coding: utf_8 -*-
import pip

# Install packages
# pip.main(["install", "-r", "requirements.txt"])

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin
from PIL import Image
from ultralytics import YOLO
from uuid import uuid4
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

def setup():
    # Load global variable from .env file
    load_dotenv()

setup()

config = Cfg.load_config_from_name("vgg_transformer") # "vgg_seq2seq"
# config = Cfg.load_config_from_name("vgg_seq2seq") # "vgg_transformer"
config["device"] = "cpu"

detector = Predictor(config)

# Start flask server
app = Flask(__name__, template_folder="./src/templates",
            static_folder="./src/static")
CORS(app)
app.config.update(
    CACHE_TYPE="null",
    CORS_HEADERS="Content-Type",
    JSON_AS_ASCII=False,
    SECRET_KEY=os.urandom(32),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    TEMPLATE_AUTO_RELOAD=True,
    UPLOAD_FOLDER="./runs/detect",
)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
@cross_origin(origins="*")
def index():
    input_image = ""
    lst = []
    detect_image = ""
    if request.method == "POST" and request.files["image"].filename:
        image = request.files["image"]
        input_image = uuid4().__str__() + "-" + image.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], input_image)
        image.save(path)
        model = YOLO("yolo_train_model.pt")
        model.predict(source=path, save=True, save_crop=True)
        detect_folder = "./runs/detect"
        index = sum(os.path.isdir(os.path.join(detect_folder, name)) for name in os.listdir(detect_folder))
        predict_folder = "./runs/detect/predict"
        predict_name = str(index) if index > 1 else ""
        drugname_folder = predict_folder + predict_name + "/crops/drugname"
        try:
            for crop in sorted(os.listdir(drugname_folder)):
                drugname = detector.predict(Image.open(drugname_folder + "/" + crop))
                drugname = drugname[drugname.find(")") + 1:].strip()
                if drugname:
                    lst.append(drugname)
        except:
            pass
        detect_image = "predict" + predict_name + "/" + os.path.split(path)[1]
    return render_template("./index.html", input_image=input_image, drugnames=lst, detect_image=detect_image)

if __name__ == "__main__":
    # Start backend server
    app.jinja_env.auto_reload = True
    app.run(host="0.0.0.0", port=int(os.getenv("PORT")))

    del app
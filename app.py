#!/usr/bin/env python3
# -*- coding: utf_8 -*-
import pip

# Install packages
# pip.main(["install", "-r", "requirements.txt"])

import base64
import cv2
import io
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Load environment variables
load_dotenv()

# Run model VietOCR to convert drugname image to text
config = Cfg.load_config_from_name("vgg_transformer") # "vgg_seq2seq"
config["device"] = "cpu"
detector = Predictor(config)

# Start flask server
app = Flask(__name__, template_folder="./src/templates", static_folder="./src/static")
CORS(app)
app.config.update(
    CACHE_TYPE="null",
    CORS_HEADERS="Content-Type",
    JSON_AS_ASCII=False,
    SECRET_KEY=os.urandom(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=True,
    TEMPLATE_AUTO_RELOAD=True,
)

# Define routes
@app.route("/", methods=["GET", "POST"])
@cross_origin(origins="*")
def index():
    # Load model
    model = YOLO("yolov8.pt")
    lst = {label: [] for label in model.names.values()}

    # Initialize variables
    classes = []
    input_image = ""
    detect_image = ""

    # Check if image is uploaded
    if request.method == "POST" and request.files["image"].filename:
        image = Image.open(request.files["image"])
        classes = [int(i) for i in request.form.getlist("classes")]

        # Convert input image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        input_image = base64.b64encode(buffered.getvalue()).decode()

        # Detect labels
        results = model.predict(classes=classes, source=image)
        boxes = results[0].boxes
        boxes = sorted(boxes, key=lambda x: x.xyxy.tolist()[0][1])

        # Crop labels image and convert to text
        for box in boxes:
            crop_region = box.xyxy.tolist()[0]
            cropped_img = image.crop(crop_region)
            label = detector.predict(cropped_img)
            if label:
                lst[model.names[int(box.cls)]].append(label)

        # Convert detect image to base64
        res_plotted = results[0].plot(conf=False, font_size=20, line_width=2, pil=True) # labels=False
        _, buffer = cv2.imencode(".png", res_plotted)
        detect_image = base64.b64encode(buffer.tobytes()).decode()

    return render_template("./index.html", classes=classes, input_image=input_image,
                           detect_image=detect_image, lst=lst)

if __name__ == "__main__":
    # Start backend server
    app.jinja_env.auto_reload = True
    app.run(host="0.0.0.0", port=int(os.getenv("PORT")))

    del app
import io
import argparse
import os
from PIL import Image

from flask import Flask, render_template, request, redirect
import torch

import mysql.connector

app = Flask(__name__)


def update_db(remote_addr, img_name, json_pred):
    rds_db = mysql.connector.connect(
    host = os.getenv('DB_HOSTNAME'),
    user = os.getenv('DB_USERNAME'),
    password = os.getenv('DB_PASSWORD'),
    database = "yolov5_predictions"
    )

    cur = rds_db.cursor()

    sql = "INSERT INTO requests (Address, ImageName, JsonPredictions) VALUES (%s, %s, %s)"
    val = (str(remote_addr), str(img_name), str(json_pred))
    cur.execute(sql, val)

    rds_db.commit()

# Load image from user
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        image_name = file.filename
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=640)
        json_pred = results.pandas().xyxy[0].to_json(orient="records")

        results.render()
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
            if 'DB_HOSTNAME' in os.environ and 'DB_USERNAME' in os.environ and 'DB_PASSWORD' in os.environ:
                update_db(request.remote_addr, image_name, json_pred)
        return redirect("static/image0.jpg")

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask web app for YOLOv5")
    parser.add_argument("--port", default=32332, type=int, help="Service port")
    parser.add_argument("--model", default="YCVR_big", type=str, help="Model to use: YCVR_big, YCVR_small, YCR_big, YCR_small, YCV")
    args = parser.parse_args()
    model_path = "./models/" + args.model + ".pt"
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, autoshape=True)
    model.eval()
    model.cpu()
    app.run(host="0.0.0.0", port=args.port)

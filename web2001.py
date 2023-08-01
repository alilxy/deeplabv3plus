import os
from PIL import Image
from flask import Flask, jsonify, request, render_template,send_file
from flask_cors import CORS
import torch
app = Flask(__name__)
CORS(app)  # 解决跨域问题
from deeplab import DeeplabV3
from werkzeug.utils import secure_filename
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_prediction(image_bytes):
    deeplab=DeeplabV3()
    file_path = r'.\static\semseg.png'
    image = Image.open(image_bytes)
    r_image = deeplab.detect_image(image, count=0, name_classes=21)
    r_image.save(file_path)
    return {"semseg": file_path}
app.config['UPLOAD_FOLDER'] =r'F:\txfg\deeplabv3-plus-pytorch-main\static'
@app.route('/upload', methods=['POST'])
def upload_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return file_path
@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path =upload_file(filename)
    print(file_path)
    info = get_prediction(image_bytes=file_path)
    return jsonify({'semseg_url': info['semseg']})
@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
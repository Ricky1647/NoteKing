from tempfile import NamedTemporaryFile
from flask import Flask, request 
from flask import Flask, render_template
from PIL import Image
import math
import json

import torch
import torch.nn  as nn
import numpy as np

from OCR.detection.model import EAST
from OCR.detection.detect import detect, crop_image, plot_boxes
from OCR.recognition.fully_conv_model import cnn_attention_ocr
from OCR.recognition.inference import make_dict

def load_models(paths, device):
	# Load Text Detection Model
	model = EAST().to(device)
	model.load_state_dict(torch.load(paths[0]))
	model.eval()

	# Load OCR model
	cnn = cnn_attention_ocr(model_dim=64,nclasses=93,n_layers=8).eval().to(device)
	cnn.load_state_dict(torch.load(paths[1])['model_state_dict'])
	decode_dict = make_dict()
	return model, cnn, decode_dict

def predict_label(img_path):
    detection_model = 'pths/east_vgg16.pth'#'/home/team1/mjtchen/EAST/EAST/pths/three_dataset_epoch100/model_epoch_10.pth'#
    OCR_model = 'pths/epoch47_iter26563.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, cnn, decode_dict = load_models([detection_model, OCR_model], device)

    img = Image.open(img_path).convert('RGB')
    #h , w , _ = np.array(img).shape
    #if (h >= w):
    #    ratio = h/512
    #    resize_ratio = tuple(map(int,(w//ratio,h/ratio)))
    #else:
    #    ratio = w/512
    #    resize_ratio = tuple(map(int,(w/ratio,h//ratio))) 
    #img = img.resize(resize_ratio, Image.BILINEAR)

    boxes = detect(img, model, device)
    imgs = crop_image(img, boxes)
    plot_img = plot_boxes(img, boxes)	
    #plot_img.save('/home/team1/yuantseng/res.bmp')
    OCR_outputs = []
    for i, pic in enumerate(imgs):
        result = OCR(pic, cnn, device, decode_dict)
        OCR_outputs.append(result)

    return boxes, plot_img, OCR_outputs

app =  Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
		return  render_template('major.html')

@app.route('/ocr_asr', methods=['GET', 'POST'])
def get_ocr_asr():
		return  render_template('index.html')

@app.route('/pureocr', methods=['GET', 'POST'])
def get_ocr():
		return  render_template('pureocr.html')

@app.route('/ocr_result', methods=['GET', 'POST'])
def get_ocr_res():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename
		img.save(img_path)
		boxes, imgs, p = predict_label(img_path)
		imgs_path = "static/test.png"
		imgs.save(imgs_path)
		#for i in boxes:
		#	print(i)
		boxes = boxes.astype(int)
		boxes = boxes.tolist()
		#print(type(boxes))
	return render_template("test1.html", prediction = p, img_path = imgs_path,boxes=boxes)
	

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		audio = request.files['my_audio']

		p = None
		img_path = None
		asr = None
		imgs_path = None
		boxes =None
		if img:
			img_path = "static/" + img.filename    
			img.save(img_path)
			boxes, imgs, p = predict_label(img_path)
			imgs_path = "static/test.png"
			imgs.save(imgs_path)
			#for i in boxes:
			#	print(i)
			boxes = boxes.astype(int)
			boxes = boxes.tolist()
		if audio:
			audio_path = "static/" + audio.filename    
			audio.save(audio_path)
			bucket_name = "cinnamon_asr"
			upload_blob(bucket_name, audio_path, "demo.wav")
			asr = transcribe_gcs("gs://"+ bucket_name+"/demo.wav")
			delete_blob(bucket_name, "demo.wav")

	return render_template("test2.html", prediction = p, img_path = imgs_path, asr = asr,boxes=boxes)

if __name__ == '__main__':
	app.run(host = '0.0.0.0',debug=True,port=5000)
from flask import Flask
from flask import request
from flask import render_template
import logging
from PIL import Image
from werkzeug.utils import secure_filename


import os
import sys
from datetime import datetime
from pathlib import Path
from io import BytesIO


from infer import infer


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


def get_img_path():
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  path = Path('static/img', timestamp)
  if not path.exists():
    path.mkdir()
  return path



@app.route('/', methods=['GET'])
def hello_world():
  return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_img():
  img = request.files['img']
  filename = img.filename
  app.logger.info('file: %s', img.filename)

  if not (img or img.filename):
    return render_template('index.html')

  img = Image.open(BytesIO(img.read()))

  if 'crop_left' in request.form:
    crop_left = float(request.form['crop_left'])
    crop_upper = float(request.form['crop_top'])
    crop_right = crop_left + float(request.form['crop_width'])
    crop_lower = crop_upper + float(request.form['crop_height'])

    img = img.crop((crop_left, crop_upper, crop_right, crop_lower))

  ext = Path(filename).suffix[1:]
  path = str(get_img_path() / filename)
  img.save(path)

  models = []
  if 'resnet' in request.form:
    models.append('resnet')
  if 'vgg' in request.form:
    models.append('vgg')
  
  res = []
  if len(models) > 0:
    res = infer(path, models)
  
  s = []
  for m, r in zip(models, res):
    s.append(m + ': ' + r)
  s = ', '.join(s)

  return render_template('index.html', result=s, img_url = '/' + path)

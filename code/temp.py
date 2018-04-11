'''
Created on Sep 29, 2016

@author: Jadiel de Armas
'''

from keras_dl.prediction.predict import Classifier
from flask import Flask, jsonify, request
import json
import signal
import sys
import imageio
import base64

app = Flask(__name__)
app.secret_key = "super secret key"

classifier = None

def signal_handler(signal, frame):
    if classifier != None:
        classifier.close_session()
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)

@app.errorhandler(Exception)
def all_exception_handler(error):
    print(error)
    return 'Error: Unsupported format', 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

@app.route('/entity/classify', methods = ['POST'])
def classify():
    if request.method == 'POST':
        stream_text = request.stream.read()
        decoded = base64.decodestring(stream_text)
        im = imageio.imread(decoded)
        klasses = classifier.classify([im])
        payload = {
            "class": klasses[0][0],
            "probability": klasses[0][1]
        }
        return str(payload)

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    with open(conf['model_conf_path']) as f:
        model_conf = json.load(f)
    labels_file = conf['labels_file']
    port = conf['port']
    
    with open(labels_file) as f:
        labels = f.read().strip().split("\n")
    
    classifier = Classifier(model_conf, labels)
    app.run(host = '0.0.0.0', port = port)
        
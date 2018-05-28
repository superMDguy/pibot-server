import numpy as np
import requests
from keras.models import load_model
from watch_gst_stream import watch_stream

model = load_model('trained.h5')

url = 'http://192.168.1.120:5000/run/'
def predict(image_arr):
    image_arr = np.expand_dims(image_arr, axis=0)
    preds = model.predict(image_arr)

    drive = np.argmax(preds[0])
    if drive == 0:
        requests.get(url + 'back')
    elif drive == 1:
        requests.get(url + 'stop_drive')
    elif drive == 2:
        requests.get(url + 'go')
    else:
        raise ValueError()

    steer = np.argmax(preds[1])
    if steer == 0:
        requests.get(url + 'left')
    elif steer == 1:
        requests.get(url + 'stop_turn')
    elif steer == 2:
        requests.get(url + 'right')
    else:
        raise ValueError()

    print(drive, steer)

watch_stream(predict, fps=10, n_frames=-1)

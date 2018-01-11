import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
Gst.init(None)
from time import sleep
import itertools


def watch_stream(callback, fps=10, n_frames=-1):
    image_arr = None

    def gst_to_numpy(sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()

        arr = np.ndarray(
            (caps.get_structure(0).get_value('height'),
            caps.get_structure(0).get_value('width'),
            3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8)
        return arr

    def new_buffer(sink, data):
        nonlocal image_arr
        sample = sink.emit('pull-sample')
        image_arr = gst_to_numpy(sample)
        return Gst.FlowReturn.OK

    pipeline=Gst.parse_launch(
        'tcpclientsrc host=192.168.1.120 port=9000 ! gdpdepay ! rtph264depay ! avdec_h264 ! videoconvert ! appsink name=sink')

    sink = pipeline.get_by_name('sink')

    sink.set_property('emit-signals', True)
    # sink.set_property('max-buffers', 2)
    # sink.set_property('drop', True)
    # sink.set_property('sync', False)

    caps=Gst.caps_from_string(
        'video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}')

    sink.set_property('caps', caps)

    sink.connect('new-sample', new_buffer, sink)

    # Start playing
    ret=pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print('Unable to set the pipeline to the playing state.')
        exit(-1)

    # Wait until error or EOS
    bus=pipeline.get_bus()

    sleep(3)  # wait for camera to warm up

    how_long = None
    if n_frames < 1:
        how_long = itertools.repeat(None) # Infinite loop
    else:
        how_long = range(n_frames)

    for _ in how_long:
        message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)

        if image_arr is not None:
            callback(image_arr)
        if message:
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(('Error received from element %s: %s' % (
                    message.src.get_name(), err)))
                print(('Debugging information: %s' % debug))
                break
            elif message.type == Gst.MessageType.EOS:
                print('End-Of-Stream reached.')
                break
            elif message.type == Gst.MessageType.STATE_CHANGED:
                if isinstance(message.src, Gst.Pipeline):
                    old_state, new_state, pending_state=message.parse_state_changed()
                    print(('Pipeline state changed from %s to %s.' %
                        (old_state.value_nick, new_state.value_nick)))
            else:
                print('Unexpected message received.')
        sleep(1 / fps)

    # Free resources
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    from PIL import Image
    from keras.applications.vgg16 import VGG16
    from keras.applications.imagenet_utils import decode_predictions, preprocess_input

    model=VGG16()

    def predict(image_arr):
        image=np.expand_dims(image_arr.copy(), axis = 0)
        image=preprocess_input(image.astype(np.float64))
        preds=model.predict(image, verbose = 0)
        print([pred[1] for pred in decode_predictions(preds, top=3)[0]])

    watch_stream(predict, fps = 10)

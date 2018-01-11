from watch_gst_stream import watch_stream

frames = []
steer = []

def append_frame(image_arr):
    frames.append(image_arr)

watch_stream(append_frame, fps=10)


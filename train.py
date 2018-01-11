from watch_gst_stream import watch_stream
import pandas as pd
import socket

frames = []
steer_cmds = []

host = '192.168.1.120'
port = 9001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))

def append_frame(image_arr):
    s.sendall(b'next')
    steer = tuple(s.recv(1024))
    print('Current controls', steer)
    steer_cmds.append(steer)
    frames.append(image_arr)

watch_stream(append_frame, fps=10, n_frames=100)
import pdb; pdb.set_trace()

s.close()
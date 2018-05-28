import socket
import pandas as pd
import numpy as np
from watch_gst_stream import watch_stream

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
try:
    watch_stream(append_frame, fps=10, n_frames=3001)
except KeyboardInterrupt:
    pass
print('Done training; saving data')

steer_df = pd.DataFrame(steer_cmds).rename(columns={0: 'drive', 1: 'steer'})
steer_df.to_csv('training_data/steer_{}.csv'.format(len(steer_df)), index=False)

np.save('training_data/frames_{}.npy'.format(len(frames)), np.array(frames))

s.close()
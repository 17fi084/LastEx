import socket
import pickle
import json

FIX_HEADER=20
#受信側
#a stream of data is coming, you should define how long of your track.
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((socket.gethostname(),8020))

while True:
    #buffer

    full_msg=b''
    new_msg=True
    while True:
        #msg=s.recv(10)
        msg=s.recv(1024)
        print(type(msg))
        print(msg)
        d=pickle.loads(msg)
        print(d)
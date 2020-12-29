import socket
import pickle
import json

FIX_HEADER=20

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
        if new_msg:
            #print('new message:{}'.format(msg[:FIX_HEADER]))
            #print('*')

            msglen=int((msg[:FIX_HEADER]))
            #print(msglen)
            new_msg=False

        full_msg+=msg
        #print(full_msg)

        if len(full_msg)-FIX_HEADER == msglen:
           #print('All Recv..')
            #print(full_msg[FIX_HEADER:])

            d=pickle.loads(full_msg[FIX_HEADER:])

            new_msg=True

            #print('*')
            print(d)
            #print(type(d))

            #json_data = json.loads(d)
            #print(json_data)

"""
# クライアントを作成
#import socket
#import joblib

#sending_data = [60,60]

#with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    #joblib.dump(sending_data, "data.jb", compress=3)
    #s.sendto(b"data.jb", ('127.0.0.1', 50007))
"""
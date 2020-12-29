import socket
import time
import pickle
import json
import joblib

#受信
#Streaming.pyと接続を試みる準備をする
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#接続待ちするIPアドレスとポートを指定
s.bind((socket.gethostname(),8020))
#接続待ち(同時に5リクエストまで処理)
print("接続待機")
s.listen(5)

#clientsocket, address = s.accept()

while True:
    # 誰かがアクセスしてきたら、コネクションとアドレスを入れる
    conn, addr = s.accept()
    print("接続中")
    with conn:
        full_msg=b''
        new_msg=True
        while True:
            #msg=s.recv(10)
            msg=conn.recv(1024)
            print(type(msg))
            print(msg)
            d=pickle.loads(msg)
            print(d)

"""
#import socket
#import joblib

#with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
   #s.bind(('127.0.0.1', 50007))

    #キーボード操作による矯正中断
    #try:
        #while True:
            #catch_data = s.recvfrom(1024)
            ##catch_data = joblib.load("data.jb") #読み出し
            #catch_data = joblib.load(catch_data)
            #print(catch_data)
            ##print("data: {}, addr: {}".format(data, addr))
            ##カメラ動作部分
            ##pantilthat.pan(50)
            ##pantilthat.tilt(50)
            ##print(pan)
            ##print(tilt)
    #except KeyboardInterrupt:
        #print("KeyboardInterrupt")
"""
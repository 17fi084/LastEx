# -*- coding: utf-8 -*-
import socket
import time
import pickle
import json
#import joblib
import pantilthat

#受信
#Streaming.pyと接続を試みる準備をする
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#接続待ちするIPアドレスとポートを指定
#s.bind((socket.gethostname(),8020))
s.bind(("192.168.1.66",8020))
print("HOSTNAME={}".format(socket.gethostname()))
#接続待ち(同時に5リクエストまで処理)
print("接続待機")
s.listen(5)

#clientsocket, address = s.accept()

while True:
    # 誰かがアクセスしてきたら、コネクションとアドレスを入れる
    conn, addr = s.accept()
    print("接続中")
    #カメラの角度
    pan_current=0
    tilt_current=0
    #カメラの限界角設定
    pan_maximum = 80
    pan_minimum = -80
    tilt_maximum = 80
    tilt_minimum = -80

    with conn:
        #full_msg=b''
        #new_msg=True
        while True:
            msg=conn.recv(1024)
            #print(type(msg))
            #print(msg)
            d=pickle.loads(msg)
            #print(d)
            json_data=d

            #if json_data["camera"]["detectdata"]["object_class"] = "NONE":
            #データを受信してカメラを動かす!
            point_X=json_data["camera"]["grid"]["point_X"]
            point_Y=json_data["camera"]["grid"]["point_Y"]

            #画像の画面に対する位置によって、カメラの方向を変える
            if point_X < 0:
                pan_current = pan_current + 1
            elif point_X > 0:
                pan_current = pan_current - 1

            if point_Y < 0:
                tilt_current = tilt_current - 1
            elif point_Y > 0:
                tilt_current = tilt_current + 1


            #カメラの限界角度を設定
            if pan_current < pan_minimum:
                pan_current = pan_minimum
            if pan_current > pan_maximum:
                pan_current = pan_maximum
            if tilt_current < tilt_minimum:
                tilt_current = tilt_minimum
            if tilt_current > tilt_maximum:
                tilt_current = tilt_maximum

            pantilthat.pan(pan_current)
            pantilthat.tilt(tilt_current)
            

            #デバッグコード
            print("point_X={}".format(point_X))
            print("point_Y={}".format(point_Y))
            print("pan_current={}".format(pan_current))
            print("tilt_current={}".format(tilt_current))

            print(json_data)


"""
コード
conda info --envs
activate yolov3
D:
cd D:\Cloud\Github\LastEx\Camera\MotionSystem
python tracking.py

"""


"""
#import socket
#import joblib
#import pantilthat

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
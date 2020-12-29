import socket
import time
import pickle
import json

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((socket.gethostname(),8020))
s.listen(5)

FIX_HEADER=20

while True:
    #jsonファイルを取得する
    file = open("TEST.json", 'r')
    #JSON形式を辞書型に変換
    json_data = json.load(file)
    #json_dataをbyte形式にしてからmsgに代入(Socketで送信するため)
    msg=pickle.dumps(json_data)

    print("json_data=")
    print(type(json_data))
    print("msg=")
    print(type(msg))
    le= str(len(msg))

    header=(((le+(FIX_HEADER-(len(str(len(msg)))))*' ')))

    header2=bytes(header.encode())
    #msg=header2+msg

    clientsocket,address=s.accept()
    #Socketでデータ送信
    print("送信データ=")
    print(type(msg))
    clientsocket.send(msg)
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
import socket
import time
import pickle
import json

#送信側
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((socket.gethostname(),8020))
s.listen(5)

while True:
    #jsonファイルを取得する
    file = open("TEST.json", 'r')
    #JSON形式を辞書型に変換
    json_data = json.load(file)
    #jsonファイルの読み出し(ここまで)


    #jsonファイルの編集
    json_data["camera"]["top"] = [50,50]

    #json_dataをbyte形式にしてからmsgに代入(Socketで送信するため)
    msg=pickle.dumps(json_data)

    #デバッグコード
    print("json_data=")
    print(type(json_data))
    print("msg=")
    print(type(msg))
    #デバッグコード(ここまで)

    clientsocket,address=s.accept()
    #Socketでデータ送信
    print("送信データ=")
    print(type(msg))
    clientsocket.send(msg)
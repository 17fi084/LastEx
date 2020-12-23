import socket
import joblib

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.bind(('127.0.0.1', 50007))

    #キーボード操作による矯正中断
    try:
        while True:
            catch_data = s.recvfrom(1024)
            #catch_data = joblib.load("data.jb") #読み出し
            catch_data = joblib.load(catch_data)
            print(catch_data)
            #print("data: {}, addr: {}".format(data, addr))
            #カメラ動作部分
            #pantilthat.pan(50)
            #pantilthat.tilt(50)
            #print(pan)
            #print(tilt)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
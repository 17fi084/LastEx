# クライアントを作成
import socket
import joblib

sending_data = [60,60]

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    joblib.dump(sending_data, "data.jb", compress=3)
    s.sendto(b"data.jb", ('127.0.0.1', 50007))
import socket
from threading import Timer
import keyboard
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
IP = "10.12.77.67"
s.bind((IP, 5000))
s.listen()
print("Server running")
sendBool = False

def ping_client():
    global sendBool
    if keyboard.is_pressed("p"):
        if not sendBool:
            message='RUN 0 0 0'
            print(message)
            clientsocket.send(bytes(message, "utf-8"))
        sendBool = True
    else:
        sendBool = False
    
  
clientsocket, address = s.accept()
print("Connections established at ", address)  
while True:
    ping_client()
    print("runnin?")
    socket_info = clientsocket.recv(1024).decode("utf-8")
    print(socket_info)
    
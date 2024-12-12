import socket

# Conecta ao servidor no localhost (127.0.0.1)
server_ip = "127.0.0.1"
port = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, port))

# Envia os dados
data = "Hello from Python!"
client_socket.sendall(data.encode('utf-8'))
client_socket.close()

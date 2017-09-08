"""
Con el modulo PiCamera se obtienen imagenes de captura
rapida para poder enviarlos a la computadora

IMPORTANTE: iniciar esto segundo para asegurar que el servidor esta escuchando
"""
import io
import socket
import struct
import subprocess
import time

import picamera

# Se crea e inicializa un zocalo de cliente para enviar los datos
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Esperando conexion de video..')
server_ip = '192.168.0.13'
if b"Fede Android" in subprocess.check_output(["iwlist", "wlan0", "scan"]):
    server_ip = '192.168.43.59'
client_socket.connect((server_ip, 8000))
print('Conexion establecida')
connection = client_socket.makefile('wb')

try:
    with picamera.PiCamera() as camera:
        # resolucion de la camara, cuadros por segundo
        camera.resolution = (320, 240)
        camera.framerate = 9
        # se duerme por 2 segundos para inicializar
        time.sleep(2)
        stream = io.BytesIO()  # envio de datos por bytes IO

        # envio de video formato JPEG
        for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
            # Enviar el tamaño de la imagen a ser envia y flushear para asegurar el envio
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            # rebobinar la imagen y enviarla como tal
            stream.seek(0)
            connection.write(stream.read())
            # situar al stream en una nueva posicion para la proxima captura
            stream.seek(0)
            stream.truncate()
    # Enviar una señal de datos igual a 0 para contar que ya se acabo el stream
    connection.write(struct.pack('<L', 0))

except IOError:
    print('Servidor del stream finalizo la conexion')
finally:
    connection.close()
    client_socket.close()

__author__ = 'federico_peralta'

"""
IMPORTANTE: iniciar esto segundo para asegurar que el servidor esta escuchando
"""
import io
import socket
import struct
import time
import picamera
from raspberry.autobot import Autobot

# Se crea e inicializa un zocalo de cliente para enviar los datos
camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Esperando conexion de la camara..')
camera_socket.connect(('192.168.0.13', 8000))
print('Conexion establecida!\nEsperando conexion del controlador del auto')
connection = camera_socket.makefile('wb')

autobot1 = Autobot(left=(27, 22), right=(10, 9))
# Se crea e inicializa un zocalo de cliente para enviar los datos
driver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
driver_socket.connect(('192.168.0.13', 8001))
print('Conexion establecida!')
autobot1.stop()

try:
    with picamera.PiCamera() as camera:
        # resolucion de la camara, cuadros por segundo
        camera.resolution = (320, 240)
        camera.framerate = 10
        # se duerme por 2 segundos para inicializar
        time.sleep(2)
        start = time.time()     # tiempo de inicio
        stream = io.BytesIO()   # envio de datos por bytes IO

        # envio de video formato JPEG
        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
            # Enviar el tamaño de la imagen a ser envia y flushear para asegurar el envio
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            # rebobinar la imagen y enviarla como tal
            stream.seek(0)
            connection.write(stream.read())
            received = driver_socket.recv(1024).decode("utf-8")
            if received == "DOF":
                autobot1.foward()
            elif received == "DOR":
                autobot1.right()
            elif received == "DOL":
                autobot1.left()
            elif received == "DOB":
                autobot1.backwards()
            elif received == "DOS":
                autobot1.stop()
            elif received == "DOE":
                driving = False
                print("Recibido comando de finalizacion...")
                break
            # si ya se establecio conexion hace mas de 600 segundos detener
            if time.time() - start > 600:
                break
            # situar al stream en una nueva posicion para la proxima captura
            stream.seek(0)
            stream.truncate()
    # Enviar una señal de datos igual a 0 para contar que ya se acabo el stream
    connection.write(struct.pack('<L', 0))

except IOError as e:
    print('Servidor del stream finalizo la conexion')
finally:
    connection.close()
    camera_socket.close()
    driver_socket.close()

__author__ = 'federico_peralta'

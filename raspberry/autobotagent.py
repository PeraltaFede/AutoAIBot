"""
IMPORTANTE: iniciar esto segundo para asegurar que el servidor esta escuchando
"""
import io
import socket
import struct
import time
import picamera
import threading
from gpiozero import Motor


class Autobot(object):
    def __init__(self, left=None, right=None):
        self.left_motor = Motor(*left)
        self.right_motor = Motor(*right)

    def foward(self, speed=1):
        self.left_motor.forward(speed)
        self.right_motor.forward(speed)

    def backwards(self, speed=1):
        self.left_motor.backward(speed)
        self.right_motor.backward(speed)

    def left(self, speed=1):
        self.left_motor.stop()
        self.right_motor.forward(speed)

    def right(self, speed=1):
        self.left_motor.forward(speed)
        self.right_motor.stop()

    def stop(self):
        self.left_motor.stop()
        self.right_motor.stop()


class VideoThread(threading.Thread):

    def __init__(self, threadid, name):
        threading.Thread.__init__(self)
        self.threadID = threadid
        self.name = name
        # Se crea e inicializa un zocalo de cliente para enviar los datos
        self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Esperando conexion de la camara..')
        self.camera_socket.connect(('192.168.0.13', 8000))
        print('Stream de la camara establecida!')
        self.connection = self.camera_socket.makefile('wb')

    def run(self):
        global running
        print("Starting " + self.name)
        try:
            with picamera.PiCamera() as camera:
                # resolucion de la camara, cuadros por segundo
                camera.resolution = (320, 240)
                camera.framerate = 10
                # se duerme por 2 segundos para inicializar
                time.sleep(2)
                start = time.time()  # tiempo de inicio
                stream = io.BytesIO()  # envio de datos por bytes IO

                # envio de video formato JPEG
                for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                    if not running:
                        break
                    # Enviar el tamaño de la imagen a ser envia y flushear para asegurar el envio
                    self.connection.write(struct.pack('<L', stream.tell()))
                    self.connection.flush()
                    # rebobinar la imagen y enviarla como tal
                    stream.seek(0)
                    self.connection.write(stream.read())
                    # si ya se establecio conexion hace mas de 600 segundos detener
                    if time.time() - start > 600:
                        break
                    # situar al stream en una nueva posicion para la proxima captura
                    stream.seek(0)
                    stream.truncate()

            # Enviar una señal de datos igual a 0 para contar que ya se acabo el stream
            self.connection.write(struct.pack('<L', 0))

        except IOError:
            print('Servidor del stream finalizo la conexion')
            print(IOError)
        finally:
            self.connection.close()
            self.camera_socket.close()


class AutobotThread(threading.Thread):

    def __init__(self, threadid, name):
        global running
        threading.Thread.__init__(self)
        self.threadID = threadid
        self.name = name
        self.autobot1 = Autobot(left=(27, 22), right=(10, 9))
        # Se crea e inicializa un zocalo de cliente para enviar los datos
        self.driver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Esperando conexion del conductor..')
        self.driver_socket.connect(('192.168.0.13', 8001))
        print('Conexion establecida!')
        self.autobot1.stop()

    def run(self):
        print("Starting " + self.name)
        try:
            while True:
                print('Esperando comandos')
                received = self.driver_socket.recv(1024).decode("utf-8")
                if received == "DOF":
                    self.autobot1.foward()
                elif received == "DOR":
                    self.autobot1.right()
                elif received == "DOL":
                    self.autobot1.left()
                elif received == "DOB":
                    self.autobot1.backwards()
                elif received == "DOS":
                    self.autobot1.stop()
                elif received == "DOE":
                    global running
                    running = False
                    print("Recibido comando de finalizacion...")
                    break

        finally:
            self.driver_socket.close()


if __name__ == '__main__':
    drivethread = AutobotThread(1, "Autobot-Thread")
    camerathread = VideoThread(2, "Camera-Thread")

    running = True
    # Start new Threads
    camerathread.start()
    drivethread.start()
    drivethread.join()
    camerathread.join()

__author__ = 'federico_peralta'

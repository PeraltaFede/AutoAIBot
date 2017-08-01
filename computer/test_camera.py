"""
inicializar este script primero, y luego abrir camera_stream en el pi
se
"""
import numpy as np
import cv2
import pygame
import socket
import struct
import os
import io
import subprocess

from pygame.locals import *

# para no confundir a pycharm y usar las librerias se debe agregar asi si no sale el autocomplete
# TODO: ELIMINAR ESTA PARTE Y TESTEAR DESDE CMD.
try:
    # noinspection PyUnresolvedReferences
    from cv2 import cv2
except ImportError:
    pass


class CameraTest(object):

    def __init__(self):

        self.server_socket = socket.socket()
        print("Inicializando stream...")
        global server_ip
        self.server_socket.bind((server_ip, 8000))
        self.server_socket.listen()
        print("Esperando conexion...")
        # bandera para el while
        self.corriendo_programa = True

        # creando conexion para enviar datos
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        print("Conexion establecida!")

        pygame.init()
        pygame.display.set_mode((50, 50), 0, 24)
        pygame.display.set_caption("Presione x o q para finalizar")
        self.open_stream()

    def open_stream(self):

        total_frame = 0
        # colecionando imagenes para el stream
        print('Iniciando streaming de la camara en: ', self.client_address)
        e1 = cv2.getTickCount()

        # obtener las imagenes del stream una por una
        try:
            while self.corriendo_programa:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    print('Finalizado por Cliente')
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write(self.connection.read(image_len))

                image_stream.seek(0)

                jpg = image_stream.read()
                image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                image = cv2.rectangle(image, (0, 120), (318, 238), (30, 230, 30), 1)

                # image = cv2.flip(image, -1)

                # guardar la imagen
                cv2.imwrite('streamtest_img/frame{:>05}.jpg'.format(total_frame), image)
                # mostrar la imagen
                cv2.imshow('Computer Vision', image)

                total_frame += 1
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        key_input = pygame.key.get_pressed()
                        if key_input[pygame.K_x] or key_input[pygame.K_q]:
                            print("Deteniendo el stream")
                            self.corriendo_programa = False
                            break
            e2 = cv2.getTickCount()
            # calcular el total de streaming
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print("Duracion del streaming:", time0)
            print('Total cuadros   : ', total_frame)
        finally:
            
            pygame.quit()
            self.connection.close()
            self.server_socket.close()
            cv2.destroyAllWindows()
            os.system("pause")

if __name__ == '__main__':
    server_ip = '192.168.0.13'
    if b"Fede Android" in subprocess.check_output("netsh wlan show interfaces"):
        server_ip = '192.168.43.59'
    CameraTest()

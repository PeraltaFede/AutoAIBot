import io
import os
import socket
import struct

import cv2
import numpy as np
import pygame
from pygame.locals import *

# para no confundir a pycharm y usar las librerias se debe agregar asi si no sale el autocomplete
# TODO: ELIMINAR ESTA PARTE Y TESTEAR DESDE CMD. debe funcionar SOLO recibiendo imagenes y enviando la direccion
try:
    # noinspection PyUnresolvedReferences
    from cv2 import cv2
except ImportError:
    pass


class AgentTrainer(object):

    def __init__(self):

        print("Iniciando stream de video, esperando conexion...")
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.0.13', 8000))
        self.server_socket.listen()
        self.connection, self.address = self.server_socket.accept()
        print("Stream de video aceptado.")
        print("Iniciando stream de control del Autobot, esperando conexion..")
        self.server2_socket = socket.socket()
        self.server2_socket.bind(('192.168.0.13', 8001))
        self.server2_socket.listen()
        self.connection2, self.address2 = self.server2_socket.accept()
        print("Autobot conectado.")

        # bandera para el while
        self.corriendo_programa = True

        pygame.init()
        self.collect_images()

    def collect_images(self):
        print("Conexion total establecida en:\nVideo  : ", self.address, "\nAutobot: ", self.address2)

        saved_frame = 0
        total_frame = 0

        # colecionando imagenes para el entrenamiento
        print('Empieza a coleccionar datos manejando.\nUtiliza las flechas '
              'para manejar. Solo se guardan los datos Arriba, Izq., Der.')
        e1 = cv2.getTickCount()

        # obtener las imagenes del stream una por una
        try:
            frame = 1
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
                image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                # region es Y, X
                roi = image[120:240, :]
                image = cv2.rectangle(image, (0, 120), (320, 240), (30, 230, 30), 2)
                # mostrar la imagen
                cv2.imshow('Computer Vision', image)

                frame += 1
                total_frame += 1
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        key_input = pygame.key.get_pressed()

                        # ordenes de dos teclas
                        if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                            print("Delante Derecha")
                            cv2.imwrite('training_images/frame{:>05}-{:>01}'.format(frame, 1), roi)
                            self.connection2.send(b"DOR")
                            saved_frame += 1

                        elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                            print("Delante Izquierda")
                            cv2.imwrite('training_images/frame{:>05}-{:>01}'.format(frame, 0), roi)
                            self.connection2.send(b"DOL")
                            saved_frame += 1

                            # ordenes una tecla
                        elif key_input[pygame.K_UP]:
                            print("Delante")
                            cv2.imwrite('training_images/frame{:>05}-{:>01}'.format(frame, 2), roi)
                            self.connection2.send(b"DOF")
                            saved_frame += 1

                        elif key_input[pygame.K_RIGHT]:
                            print("Derecha")
                            cv2.imwrite('training_images/frame{:>05}-{:>01}'.format(frame, 1), roi)
                            self.connection2.send(b"DOR")
                            saved_frame += 1

                        elif key_input[pygame.K_LEFT]:
                            print("Izquierda")
                            cv2.imwrite('training_images/frame{:>05}-{:>01}'.format(frame, 0), roi)
                            self.connection2.send(b"DOL")
                            saved_frame += 1

                        elif key_input[pygame.K_DOWN]:
                            self.connection2.send(b"DOB")
                            print("Reversa")

                        elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                            print("Detener el programa")
                            self.connection2.send(b"DOE")
                            self.corriendo_programa = False
                            break

                    elif event.type == pygame.KEYUP:

                        key_input = pygame.key.get_pressed()

                        if key_input[pygame.K_UP]:
                            print("Delante")
                            cv2.imwrite('training_images/frame{:>05}-{:>01}'.format(frame, 2), roi)
                            self.connection2.send(b"DOF")
                            saved_frame += 1

                        elif key_input[pygame.K_RIGHT]:
                            print("Derecha")
                            cv2.imwrite('training_images/frame{:>05}-{:>01}'.format(frame, 1), roi)
                            self.connection2.send(b"DOR")
                            saved_frame += 1

                        elif key_input[pygame.K_LEFT]:
                            print("Izquierda")
                            cv2.imwrite('training_images/frame{:>05}-{:>01}'.format(frame, 0), roi)
                            self.connection2.send(b"DOL")
                            saved_frame += 1

                        elif key_input[pygame.K_DOWN]:
                            self.connection2.send(b"DOB")
                            print("Reversa")

                        else:
                            self.connection2.send(b"DOS")
                            print('Esperando ordenes')

            e2 = cv2.getTickCount()
            # calcular el total de streaming
            time0 = (e2 - e1) / cv2.getTickFrequency()
            pygame.quit()
            cv2.destroyAllWindows()
            print("Duracion del streaming:", time0)
            print('Total cuadros           : ', total_frame)
            print('Total cuadros guardados : ', saved_frame)
            print('Total cuadros desechados: ', total_frame - saved_frame)
            os.system('pause')
        finally:
            self.connection.close()
            self.server_socket.close()
            self.connection2.close()
            self.server2_socket.close()

if __name__ == '__main__':
    AgentTrainer()

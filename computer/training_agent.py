import numpy as np
import cv2
import pygame
import socket

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

        print("Iniciando stream")
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.0.9', 8000))
        self.server_socket.listen(1)

        # bandera para el while
        self.corriendo_programa = True

        # creando conexion para enviar datos
        print("Esperando conexion...")
        self.connection, self.address = self.server_socket.accept()  # [0].makefile('rb')

        # creando etiquetas
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')

        pygame.init()
        self.collect_image()

    def collect_image(self):
        print("Conexion establecida!")

        saved_frame = 0
        total_frame = 0

        # colecionando imagenes para el entrenamiento
        print('Empieza a coleccionar datos manejando.\nUtiliza las flechas '
              'para manejar. Solo se guardan los datos Arriba, Izq., Der.')
        e1 = cv2.getTickCount()
#        image_array = np.zeros((1, 38400))
#        label_array = np.zeros((1, 4), 'float')

        # obtener las imagenes del stream una por una
        try:
            stream_bytes = ' '
            frame = 1
            while self.corriendo_programa:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

                    # la parte que quiero de toda la imagen Region Of Interest
                    roi = image[0:200, :]

                    # guardar la imagen
                    cv2.imwrite('captured_images/frame{:>05}.jpg'.format(frame), image)
                    # mostrar la imagen
                    cv2.imshow('Computer Vision', image)

                    # cambiar la imagen ROI a un vector
#                    temp_array = roi.reshape(1, 38400).astype(np.float32)

                    frame += 1
                    total_frame += 1
                    for event in pygame.event.get():
                        if event.type == KEYDOWN:
                            key_input = pygame.key.get_pressed()

                            # ordenes de dos teclas
                            if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                                print("Delante Derecha")
                                cv2.imwrite('training_images/frame{:>05}-{:>05}'.format(frame, 1), roi)
#                                image_array = np.vstack((image_array, temp_array))
#                                label_array = np.vstack((label_array, self.k[1]))
                                self.connection.send("DOR")
                                saved_frame += 1

                            elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                                print("Delante Izquierda")
                                cv2.imwrite('training_images/frame{:>05}-{:>05}'.format(frame, 0), roi)
#                                image_array = np.vstack((image_array, temp_array))
#                                label_array = np.vstack((label_array, self.k[0]))
                                self.connection.send("DOL")
                                saved_frame += 1

                            # ordenes una tecla
                            elif key_input[pygame.K_UP]:
                                print("Delante")
                                cv2.imwrite('training_images/frame{:>05}-{:>05}'.format(frame, 2), roi)
#                                image_array = np.vstack((image_array, temp_array))
#                                label_array = np.vstack((label_array, self.k[2]))
                                self.connection.send("DOF")
                                saved_frame += 1

                            elif key_input[pygame.K_RIGHT]:
                                print("Derecha")
                                cv2.imwrite('training_images/frame{:>05}-{:>05}'.format(frame, 1), roi)
#                                image_array = np.vstack((image_array, temp_array))
#                                label_array = np.vstack((label_array, self.k[1]))
                                self.connection.send("DOR")
                                saved_frame += 1

                            elif key_input[pygame.K_LEFT]:
                                print("Izquierda")
                                cv2.imwrite('training_images/frame{:>05}-{:>05}'.format(frame, 0), roi)
#                                image_array = np.vstack((image_array, temp_array))
#                                label_array = np.vstack((label_array, self.k[0]))
                                self.connection.send("DOL")
                                saved_frame += 1

                            elif key_input[pygame.K_DOWN]:
                                self.connection.send("DOB")
                                print("Reversa")

                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print("Detener el programa")
                                self.connection.send("DOE")
                                self.corriendo_programa = False
                                break

                        elif event.type == pygame.KEYUP:

                            key_input = pygame.key.get_pressed()

                            if key_input[pygame.K_UP]:
                                print("Delante")
                                cv2.imwrite('training_images/frame{:>05}-{:>05}'.format(frame, 2), roi)
#                                image_array = np.vstack((image_array, temp_array))
#                                label_array = np.vstack((label_array, self.k[2]))
                                self.connection.send("DOF")
                                saved_frame += 1

                            elif key_input[pygame.K_RIGHT]:
                                print("Derecha")
                                cv2.imwrite('training_images/frame{:>05}-{:>05}'.format(frame, 1), roi)
#                                image_array = np.vstack((image_array, temp_array))
#                                label_array = np.vstack((label_array, self.k[1]))
                                self.connection.send("DOR")
                                saved_frame += 1

                            elif key_input[pygame.K_LEFT]:
                                print("Izquierda")
                                cv2.imwrite('training_images/frame{:>05}-{:>05}'.format(frame, 0), roi)
#                                image_array = np.vstack((image_array, temp_array))
#                                label_array = np.vstack((label_array, self.k[0]))
                                self.connection.send("DOL")
                                saved_frame += 1

                            elif key_input[pygame.K_DOWN]:
                                self.connection.send("DOB")
                                print("Reversa")

                            else:
                                self.connection.send("DOS")
                                print('Esperando ordenes')

            # guardar las imagenes y sus etiquetas
#            train = image_array[1:, :]
#            train_labels = label_array[1:, :]

            # guardar los datos de entrenamiento como ficheros numpy
#            file_name = str(int(time.time()))
#            directory = "training_data"
#            if not os.path.exists(directory):
#                os.makedirs(directory)
#            try:
#                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
#            except IOError as e:
#                print(e)

            e2 = cv2.getTickCount()
            # calcular el total de streaming
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print("Duracion del streaming:", time0)
#            print(train.shape)
#            print(train_labels.shape)
            print('Total cuadros           : ', total_frame)
            print('Total cuadros guardados : ', saved_frame)
            print('Total cuadros desechados: ', total_frame - saved_frame)
        finally:
            self.connection.close()
            self.server_socket.close()

if __name__ == '__main__':
    AgentTrainer()

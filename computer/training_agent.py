import io
import struct
import threading
# noinspection PyCompatibility
import socketserver
import os
import subprocess
import random

import cv2
import numpy as np
import pygame


# para no confundir a pycharm y usar las librerias se debe agregar asi si no sale el autocomplete
# TODO: ELIMINAR ESTA PARTE Y TESTEAR DESDE CMD. debe funcionar SOLO recibiendo imagenes y enviando la direccion
try:
    # noinspection PyUnresolvedReferences
    from cv2 import cv2
except ImportError:
    pass


class AutobotThread(socketserver.StreamRequestHandler):

    def handle(self):
        pygame.init()
        myfont = pygame.font.SysFont("monospace", 15)
        screen = pygame.display.set_mode((200, 200), 0, 24)
        label = myfont.render("Detenido", 1, (255, 255, 0))
        screen.blit(label, (0, 60))
        pygame.display.flip()

        print("Conexion establecida en Autobot: ", self.client_address)
        print('Empieza a coleccionar datos manejando.\nUtiliza las flechas '
              'para manejar. Solo se guardan los datos Arriba, Izq., Der.')

        try:
            global running, saved_frame, roi, newimg
            saved_frame = 0
            currentstate = 4  # 0 = izquierda ; 1 = derecha; 2 = delante ; 3 = reversa; 4 = stop
            while running:
                if newimg:
                    newimg = False
                    cv2.imshow('Computer vision', realimg)
                    key_input = pygame.key.get_pressed()
                    # ordenes de dos teclas
                    if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                        if random.randint(0, 99) > 15:
                            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 1), roi)
                        else:
                            cv2.imwrite('test_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 1), roi)
                        if not currentstate == 1:
                            self.connection.send(b"DOR")
                            currentstate = 1
                            label = myfont.render("Delante Derecha", 1, (255, 255, 0), (0, 0, 0))
                        saved_frame += 1

                    elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                        if random.randint(0, 99) > 15:
                            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 0), roi)
                        else:
                            cv2.imwrite('test_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 0), roi)
                        if not currentstate == 0:
                            self.connection.send(b"DOL")
                            currentstate = 0
                            label = myfont.render("Delante Izquierda", 1, (255, 255, 0), (0, 0, 0))
                        saved_frame += 1

                        # ordenes una tecla
                    elif key_input[pygame.K_UP]:
                        if random.randint(0, 99) > 15:
                            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 2), roi)
                        else:
                            cv2.imwrite('test_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 2), roi)
                        if not currentstate == 2:
                            self.connection.send(b"DOF")
                            currentstate = 2
                            label = myfont.render("Delante", 1, (255, 255, 0), (0, 0, 0))
                        saved_frame += 1

                    elif key_input[pygame.K_RIGHT]:
                        if random.randint(0, 99) > 15:
                            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 1), roi)
                        else:
                            cv2.imwrite('test_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 1), roi)
                        if not currentstate == 1:
                            self.connection.send(b"DOR")
                            currentstate = 1
                            label = myfont.render("Derecha", 1, (255, 255, 0), (0, 0, 0))
                        saved_frame += 1

                    elif key_input[pygame.K_LEFT]:
                        if random.randint(0, 99) > 15:
                            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 0), roi)
                        else:
                            cv2.imwrite('test_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 0), roi)
                        if not currentstate == 0:
                            self.connection.send(b"DOL")
                            currentstate = 0
                            label = myfont.render("Izquierda", 1, (255, 255, 0), (0, 0, 0))
                        saved_frame += 1

                    elif key_input[pygame.K_DOWN]:
                        if not currentstate == 3:
                            self.connection.send(b"DOB")
                            currentstate = 3
                            label = myfont.render("Reversa", 1, (255, 255, 0), (0, 0, 0))

                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print("Detener el programa")
                        self.connection.send(b"DOE")
                        running = False
                        break

                    else:
                        if not currentstate == 4:
                            label = myfont.render("Detenido", 1, (255, 255, 0), (0, 0, 0))
                            currentstate = 4
                            self.connection.send(b"DOS")
                    screen.blit(label, (0, 60))
                    screen.blit(myfont.render(("Total Frames: " + str(total_frame)),
                                              1, (255, 255, 0), (0, 0, 0)), (0, 0))
                    screen.blit(myfont.render(("Saved Frames: " + str(saved_frame)),
                                              1, (255, 255, 0), (0, 0, 0)), (0, 30))
                    pygame.display.flip()
                else:
                    for _ in pygame.event.get():
                        _ = pygame.key.get_pressed()

            pygame.quit()
            cv2.destroyAllWindows()
        finally:
            print('Server finalizado en AutobotDriver')


class VideoThread(socketserver.StreamRequestHandler):

    name = "Video-Thread"

    def handle(self):
        global running, roi, total_frame, realimg, newimg
        total_frame = 0
        print("Conexion establecida video: ", self.client_address)
        running = True
        roi = 0
        # obtener las imagenes del stream una por una
        try:
            while running:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_len = struct.unpack('<L', self.rfile.read(struct.calcsize('<L')))[0]
                if not image_len:
                    print('Finalizado por Cliente')
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection

                image_stream = io.BytesIO()
                image_stream.write(self.rfile.read(image_len))

                image_stream.seek(0)
                jpg = image_stream.read()
                realimg = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                # region es Y, X
                roi = image[120:240, :]
                realimg = cv2.rectangle(realimg, (0, 120), (318, 238), (30, 230, 30), 1)
                newimg = True
                total_frame += 1
        finally:
            print('Server finalizado en VideoStreaming')


class ThreadServer(object):

    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), AutobotThread)
        server.serve_forever()

    def server_thread2(host, port):
        server = socketserver.TCPServer((host, port), VideoThread)
        server.serve_forever()

    server_ip = '192.168.0.13'
    if b"Fede Android" in subprocess.check_output("netsh wlan show interfaces"):
        server_ip = '192.168.43.59'
    print(server_ip)
    video_thread = threading.Thread(target=server_thread2, args=(server_ip, 8000))
    video_thread.start()
    print("Video thread started")
    autobot_thread = threading.Thread(target=server_thread, args=(server_ip, 8001))
    autobot_thread.start()
    print("Autobot thread started")


if __name__ == '__main__':

    running = True
    saved_frame = 0
    total_frame = 0
    roi = None
    realimg = None
    newimg = False
    e1 = cv2.getTickCount()
    ThreadServer()
    while running:
        pass
    e2 = cv2.getTickCount()
    # calcular el total de streaming
    time0 = (e2 - e1) / cv2.getTickFrequency()
    print("Duracion del streaming:", time0)
    print('Total cuadros           : ', total_frame)
    print('Total cuadros guardados : ', saved_frame)
    print('Total cuadros desechados: ', total_frame - saved_frame)
    os.system('exit()')

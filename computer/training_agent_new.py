import io
import socket
import os
# noinspection PyCompatibility
import struct
import subprocess

import cv2
import numpy as np
import pygame

server_ip = '192.168.0.13'
if b"Fede Android" in subprocess.check_output("netsh wlan show interfaces"):
    server_ip = '192.168.43.59'

print("Inicializando stream...")

server_video_socket = socket.socket()
server_video_socket.bind((server_ip, 8000))
print("Esperando conexion de video, inicie ahora camera_stream.py en el AutoBot...")
server_video_socket.listen()
video_connection, client_video_address = server_video_socket.accept()
video_connection = video_connection.makefile('rb')
print("Conexion establecida de video en", client_video_address)

server_control_socket = socket.socket()
server_control_socket.bind((server_ip, 8001))
print("Esperando conexion de controlador del autobot, inicie ahora autobot.py en el AutoBot...")
server_control_socket.listen()
# creando conexion para enviar datos
control_connection, client_control_address = server_control_socket.accept()
print("Conexion establecida de video en", client_control_address)

pygame.init()
# bandera para el while
running = True
saved_frame = 0
total_frame = 0
e1 = cv2.getTickCount()

try:
    myfont = pygame.font.SysFont("monospace", 15)
    screen = pygame.display.set_mode((200, 200), 0, 24)
    label = myfont.render("Empieza a coleccionar datos manejando.\nUtiliza las flechas" +
                          "para manejar. Solo se guardan los datos Arriba, Izq., Der.", 1, (255, 255, 0))
    screen.blit(label, (0, 0))
    pygame.display.flip()
    saved_frame = 0
    currentstate = 4  # 0 = izquierda ; 1 = derecha; 2 = delante ; 3 = reversa; 4 = stop

    while running:
        a1 = cv2.getTickCount()
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', video_connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            print('Finalizado por Cliente')
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(video_connection.read(image_len))

        image_stream.seek(0)

        jpg = image_stream.read()
        roi = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # region es Y, X
        roi = roi[120:240, :]
        # mostrar la imagen
        cv2.imshow('Computer Vision', roi)
        cv2.imwrite('frame', roi)
        total_frame += 1
        key_input = pygame.key.get_pressed()
        # ordenes de dos teclas
        if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 1), roi)
            if not currentstate == 1:
                control_connection.send(b"DOR")
                currentstate = 1
                label = myfont.render("Delante Derecha", 1, (255, 255, 0))
            saved_frame += 1

        elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 0), roi)
            if not currentstate == 0:
                control_connection.send(b"DOL")
                currentstate = 0
                label = myfont.render("Delante Izquierda", 1, (255, 255, 0))
            saved_frame += 1

            # ordenes una tecla
        elif key_input[pygame.K_UP]:
            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 2), roi)
            if not currentstate == 2:
                control_connection.send(b"DOF")
                currentstate = 2
                label = myfont.render("Delante", 1, (255, 255, 0))
            saved_frame += 1

        elif key_input[pygame.K_RIGHT]:
            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 1), roi)
            if not currentstate == 1:
                control_connection.send(b"DOR")
                currentstate = 1
                label = myfont.render("Derecha", 1, (255, 255, 0))
            saved_frame += 1

        elif key_input[pygame.K_LEFT]:
            cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 0), roi)
            if not currentstate == 0:
                control_connection.send(b"DOL")
                currentstate = 0
                label = myfont.render("Izquierda", 1, (255, 255, 0))
            saved_frame += 1

        elif key_input[pygame.K_DOWN]:
            if not currentstate == 3:
                control_connection.send(b"DOB")
                currentstate = 3
                label = myfont.render("Reversa", 1, (255, 255, 0))

        elif key_input[pygame.K_x] or key_input[pygame.K_q]:
            print("Detener el programa")
            control_connection.send(b"DOE")
            running = False
            break

        else:
            if not currentstate == 4:
                label = myfont.render("Detenido", 1, (255, 255, 0))
                currentstate = 4
                control_connection.send(b"DOS")

        for _ in pygame.event.get():
            _ = pygame.key.get_pressed()

        screen.fill((0, 0, 0))
        screen.blit(label, (0, 60))
        screen.blit(myfont.render(("Total Frames: " + str(total_frame)),
                                  1, (255, 255, 0)), (0, 0))
        screen.blit(myfont.render(("Saved Frames: " + str(saved_frame)),
                                  1, (255, 255, 0)), (0, 30))
        pygame.display.flip()
        a2 = cv2.getTickCount()
        time1 = (a2-a1) / cv2.getTickFrequency()
        print(time1)


finally:

    pygame.quit()
    video_connection.close()
    server_video_socket.close()
    control_connection.close()
    server_control_socket.close()
    cv2.destroyAllWindows()
    e2 = cv2.getTickCount()
    time0 = (e2 - e1) / cv2.getTickFrequency()
    print("Duracion del streaming:", time0)
    print('Total cuadros           : ', total_frame)
    print('Total cuadros guardados : ', saved_frame)
    print('Total cuadros desechados: ', total_frame - saved_frame)
    os.system("pause")  # calcular el total de streaming
    os.system('exit()')

import os
import socket
import subprocess

import pygame
from pygame.locals import *


class ABTest(object):
    def __init__(self):

        print("Iniciando modulo prueba de autobot")
        self.server_socket = socket.socket()
        global server_ip
        self.server_socket.bind((server_ip, 8001))
        self.server_socket.listen()

        print("Esperando conexion...")
        self.test_drive = True

        self.connection, self.client_addres = self.server_socket.accept()
        # self.connection = self.connection.makefile('rb')

        pygame.init()
        self.steer()

    def steer(self):
        try:
            myfont = pygame.font.SysFont("monospace", 15)
            screen = pygame.display.set_mode((200, 200), 0, 24)
            label = myfont.render("Detenido", 1, (255, 255, 0))
            screen.blit(label, (0, 0))
            pygame.display.flip()

            print("Modulo conectado, autobot conectado en: ", self.client_addres)
            print('Presione las flechas para mover el autobot...')
            # siempre se busca un evento del teclado, si alguna tecla esta apretada, se verifica cual y se mueve el auto
            while self.test_drive:
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        key_input = pygame.key.get_pressed()

                        # complex orders
                        if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                            self.connection.send(b"DOR")
                            label = myfont.render("Delante Derecha", 1, (255, 255, 0))

                        elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                            self.connection.send(b"DOL")
                            label = myfont.render("Delante Izquierda", 1, (255, 255, 0))

                        # simple orders
                        elif key_input[pygame.K_UP]:
                            self.connection.send(b"DOF")
                            label = myfont.render("Delante", 1, (255, 255, 0))

                        elif key_input[pygame.K_DOWN]:
                            self.connection.send(b"DOB")
                            label = myfont.render("Reversa", 1, (255, 255, 0))

                        elif key_input[pygame.K_RIGHT]:
                            self.connection.send(b"DOR")
                            label = myfont.render("Derecha", 1, (255, 255, 0))

                        elif key_input[pygame.K_LEFT]:
                            self.connection.send(b"DOL")
                            label = myfont.render("Izquierda", 1, (255, 255, 0))

                        elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                            print('Detener el programa')
                            self.connection.send(b"DOE")
                            self.test_drive = False
                            break

                    elif event.type == pygame.KEYUP:
                        key_input = pygame.key.get_pressed()

                        if key_input[pygame.K_UP]:
                            self.connection.send(b"DOF")
                            label = myfont.render("Delante", 1, (255, 255, 0))

                        elif key_input[pygame.K_DOWN]:
                            self.connection.send(b"DOB")
                            label = myfont.render("Reversa", 1, (255, 255, 0))

                        elif key_input[pygame.K_RIGHT]:
                            self.connection.send(b"DOR")
                            label = myfont.render("Derecha", 1, (255, 255, 0))

                        elif key_input[pygame.K_LEFT]:
                            self.connection.send(b"DOL")
                            label = myfont.render("Izquierda", 1, (255, 255, 0))

                        else:
                            self.connection.send(b"DOS")
                            label = myfont.render("Detenido", 1, (255, 255, 0))

                screen.fill((0, 0, 0))
                screen.blit(label, (0, 0))
                pygame.display.flip()

            print('Test drive finalizado')
            pygame.quit()
        finally:
            self.connection.close()
            self.server_socket.close()


# se pone asi para que el script no corra cuando sea importado
if __name__ == '__main__':
    server_ip = '192.168.0.13'
    if b"Fede Android" in subprocess.check_output("netsh wlan show interfaces"):
        server_ip = '192.168.43.59'
    ABTest()
    os.system("pause")

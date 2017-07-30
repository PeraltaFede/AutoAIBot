import pygame
import os
import socket
import subprocess
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
        pygame.display.set_mode((20, 20), 0, 24)
        pygame.display.set_caption("Teclado")

        self.steer()

    def steer(self):
        try:
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
                            print("Delante derecha")

                        elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                            self.connection.send(b"DOL")
                            print("Delante izquierda")

                        # simple orders
                        elif key_input[pygame.K_UP]:
                            self.connection.send(b"DOF")
                            print("Delante")

                        elif key_input[pygame.K_DOWN]:
                            self.connection.send(b"DOB")
                            print("Reversa")

                        elif key_input[pygame.K_RIGHT]:
                            self.connection.send(b"DOR")
                            print("Derecha")

                        elif key_input[pygame.K_LEFT]:
                            self.connection.send(b"DOL")
                            print("Izquierda")
                        elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                            print('Salir')
                            self.connection.send(b"DOE")
                            self.test_drive = False
                            break

                    elif event.type == pygame.KEYUP:
                        key_input = pygame.key.get_pressed()

                        if key_input[pygame.K_UP]:
                            self.connection.send(b"DOF")
                            print("Delante")

                        elif key_input[pygame.K_DOWN]:
                            self.connection.send(b"DOB")
                            print("Reversa")

                        elif key_input[pygame.K_RIGHT]:
                            self.connection.send(b"DOR")
                            print("Derecha")

                        elif key_input[pygame.K_LEFT]:
                            self.connection.send(b"DOL")
                            print("Izquierda")
                        else:
                            self.connection.send(b"DOS")
                            print('Esperando ordenes')

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

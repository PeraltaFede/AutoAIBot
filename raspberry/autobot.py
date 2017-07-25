"""
Con el auto montado se envia informacion de listo CAMBIAR A NINGUNA PARA EL FINAL
y se leen los datos entrantes para mover el auto
IMPORTANTE: iniciar esto segundo para asegurar que el servidor esta escuchando
"""
import socket
from gpiozero import Motor


class Autobot(object):
    def __init__(self, left=None, right=None):
        self.left_motor = Motor(*left)
        self.right_motor = Motor(*right)

    def forward(self, speed=1):
        self.left_motor.forward(speed)
        self.right_motor.forward(speed)

    def backwards(self, speed=1):
        self.left_motor.backward(speed)
        self.right_motor.backward(speed)

    def left(self, speed=1):
        self.left_motor.forward(speed)
        self.right_motor.forward(speed/2)

    def right(self, speed=1):
        self.left_motor.forward(speed/2)
        self.right_motor.forward(speed)

    def stop(self):
        self.left_motor.stop()
        self.right_motor.stop()

if __name__ == '__main__':

    # TODO: Cambiar aca para mover bien
    autobot1 = Autobot(left=(22, 27), right=(9, 10))
    autobot1.stop()
    # Se crea e inicializa un zocalo de cliente para enviar los datos
    print('Esperando conexion del autobot..')
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.0.13', 8001))
    print('Conexion del autobot establecida!')

    try:
        driving = True
        while driving:
            received = client_socket.recv(1024).decode("utf-8")
            if received == "DOF":
                autobot1.forward()
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
                autobot1.stop()
                print("Recibido comando de finalizacion...")
                break

    finally:
        client_socket.close()

__author__ = 'federico_peralta'

#probes all sensors and prints its values client side
import time
import struct
import numpy as np
from libs.Quanser.product_QCar import QCar
from libs.Quanser.q_ui import gamepadViaTarget
from libs.Quanser.q_misc import Calculus
from libs.Quanser.q_interpretation import basic_speed_estimation
from libs.stream_struct import StructStream


car = QCar()

sampleRate = 100.0
sampleTime = 1/sampleRate
#simulationTime = 30.0

dataPacket = struct.Struct("7f")
bufferSize = dataPacket.size
mtr_cmd = np.array([0.0,0.0])
diff = Calculus().differentiator_variable(sampleTime)
_ = next(diff)
timeStep = sampleTime
count = 0

gpad = gamepadViaTarget(5)
#configuration = '3'

serverIP = "tcpip://192.168.2.11:18001"
client = StructStream(serverIP, 'c', bufferSize, bufferSize)
prevCon = False

startTime = time.time()
def elapsed_time():
    return time.time() - startTime

def spikeLinInject(sensor1, fso, count):
    '''injects errors into the sensor data to test detection AI.'''
    error = np.random.randint(100)
    gradient = 0.01
    if error == 0:
        sigval = np.average(sensor1)
        sensor1 += 0.5*gradient*fso*count
        count += 1
    return sensor1, count
try:
    while True:
        if not client.connected:
            client.checkConnection()

        if client.connected and not prevCon:
            print("Connection to server succesfully established!")
            prevCon = client.connected

        elif client.connected:
            sinceStart = elapsed_time()
            start = time.time()

            new = gpad.read()

            if new and gpad.LT:
                mtr_cmd = np.array([0.3*gpad.RLO, 0.5*gpad.LLA])
            LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
            
            car.read_write_std(mtr_cmd, LEDs)

            gyroscope = car.read_gyroscope()
            accelerometer = car.read_accelerometer()
            encoder = car.read_encoder()
            accelerometer[2] -= 9.8 #subtracts the acceleration due to gravity
            encoderSpeed = diff.send((encoder, timeStep))
            linearSpeed = basic_speed_estimation(encoderSpeed)
            if sinceStart >= 100.0:
                linearSpeed, count = spikeLinInject(linearSpeed, 20, count)
            buffer = dataPacket.pack(gyroscope[0], gyroscope[1], gyroscope[2], accelerometer[0], accelerometer[1], accelerometer[2], linearSpeed)
            client.send(buffer)

            end = time.time()
            computationTime = end - start
            sleepTime = sampleTime - (computationTime % sampleTime)
            #print("Speed: ", linearSpeed)
            #print('Gyroscope: ', gyroscope, ' accelerometer: ', accelerometer, ' encoder: ', encoder)
            time.sleep(sleepTime)
        
except KeyboardInterrupt:
    print("Keyboard Interrupt!")

finally:
    car.terminate()
    client.terminate()
    print("Client Terminated")
        

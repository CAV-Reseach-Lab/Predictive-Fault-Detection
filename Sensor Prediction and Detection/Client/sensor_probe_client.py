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
mtr_cmd = np.array([0.0, 0.0])
diff = Calculus().differentiator_variable(sampleTime)
_ = next(diff)
timeStep = sampleTime
count = 0
startErr = 0
stuckVal = 0

gpad = gamepadViaTarget(5)
#configuration = '3'

serverIP = "tcpip://192.168.2.11:18001"
client = StructStream(serverIP, 'c', bufferSize, bufferSize)
prevCon = False

startTime = time.time()


def elapsed_time():
    return time.time() - startTime


def discombobulate(sensor1, sensor2, sensor3, error, typeError, count, whatSensor, stuckVal):
    '''injects errors into the sensor data to test detection AI.'''
    #error = np.random.randint(10000)
    if error == 1:
        #print("FAULT!")
        #typeError = np.random.randint(3)
        if typeError == 0:
            #erratic
            injectError = np.random.normal(0, 0.2)
        elif typeError == 1:
            #spike
            injectError = np.random.randint(low=5, high=10)
        elif typeError == 2:
            driftGrad = 0.005
            injectError = driftGrad * count + 0.1
            count += 1
        elif typeError == 3:
            #stuck
            if count == 0:
                if whatSensor == 0:
                    stuckVal = sensor1[0]
                elif whatSensor == 1:
                    stuckVal = sensor2[0]
                elif whatSensor == 3:
                    stuckVal = sensor3
                injectError = stuckVal
                count += 1
            elif count >= 1:
                injectError = stuckVal
        else:
            #hardover
            injectError = 20  # idk
        #whatSensor = 0
        #np.random.randint(3)7
        if whatSensor == 0:
            sensor1 += injectError
        #if whatSensor == 0:
        #    sensor1[0] = injectError
        #    sensor1[1] = injectError
        #    sensor1[2] = injectError
        elif whatSensor == 1:
            sensor2 += injectError
        else:
            sensor3 += injectError
    return sensor1, sensor2, sensor3


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
            #if startErr == 0:
            #   startErr = np.random.randint(2)
            #   errDur = np.random.randint(60)
            #    durStart = time.time()
            #    count = 0

            new = gpad.read()

            if new and gpad.LT:
                mtr_cmd = np.array([0.3*gpad.RLO, 0.5*gpad.LLA])
            LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
            car.read_write_std(mtr_cmd, LEDs)

            gyroscope = car.read_gyroscope()
            accelerometer = car.read_accelerometer()
            encoder = car.read_encoder()
            # subtracts the acceleration due to gravity
            accelerometer[2] -= 9.8
            if sinceStart >= 120.0:
                gyroscope, accelerometer, encoder = discombobulate(
                    gyroscope, accelerometer, encoder, 1, 2, count, 0, stuckVal)
            #if startErr == 1:
             #   dur = time.time()
             #   if (dur - durStart) <= errDur:
             #       gyroscope, accelerometer, encoder = discombobulate(gyroscope, accelerometer, encoder,1,3,count,0, stuckVal)
             #   if (dur - durStart) > errDur:
             #       startErr = 0
            encoderSpeed = diff.send((encoder, timeStep))
            linearSpeed = basic_speed_estimation(encoderSpeed)
            buffer = dataPacket.pack(gyroscope[0], gyroscope[1], gyroscope[2],
                                     accelerometer[0], accelerometer[1], accelerometer[2], linearSpeed)
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

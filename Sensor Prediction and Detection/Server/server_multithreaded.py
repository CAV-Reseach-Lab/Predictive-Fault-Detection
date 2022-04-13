import multiprocessing as mp
import sensor_probe_server as server
import detector_net as detector
#import sampleData

def dataSample(q):
    '''Samples the incoming data and returns a 240 samples of data worth about 2.4 seconds of sensor data"'''
    dataFrame = np.array([[0,0,0]], dtype = np.float32)
    fieldsName = ['Gyroscope', 'Accelerometer', 'Encoder']
    i = 0
    for i in range(239):
        data[0], data[1], data[2] = myServer.recieve()
        avgGyro = (data[0][0] + data[0][1] + data[0][2])/3.0
        avgAcc = (data[1][0] + data[1][1] + data[1][2])/3.0
        tmpArr = [avgGyro, avgAcc, data[2]]
        #with open("./sampleData.csv", 'a') as csvFile:
        #    writer = csv.DictWriter(csvFile, fieldsName)
        #    writer.writerow({'Gyroscope': tmpArr[0], 'Accelerometer': tmpArr[1], 'Encoder': tmpArr[2]})
        dataFrame = np.vstack((dataFrame, tmpArr))
    dataFrame = np.reshape(dataFrame,(1, 3, 240)) 
    tensor = from_numpy(dataFrame)
    return tensor

def main():
    myServer = server.server('192.168.2.12')
    connected = False
    model = detector.detectorNet()
    q = Queue()
    try:
        while not myServer.server.connected:
            print("Server initialization complete. Ready to recieve data.")
            connected = myServer.connect()
            if not connected:
                print("Trying again")
        while myServer.server.connected:
            #data[0], data[1], data[2] = myServer.recieve()
            #print("Gyroscope: ", data[0], " Acccelerometer: ", data[1], " Encoder: ", data[2])
            dataTensor = dataSample.dataSample(q)
            result, HI = model.predict(dataTensor)
            print("Result: ", result, " HI: ", HI)
    except KeyboardInterrupt:
        print("User terminated the server!")
    finally:
        myServer.terminate()

if __name__ == '__main__':
    main()
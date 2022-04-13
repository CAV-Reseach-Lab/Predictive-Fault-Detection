#This set of python code is like the main function. This is where the program will start.
import time
import queue as Queue
import threading
import numpy as np
import pandas as pd
import sensor_probe_server as server
import detector_net as detector
import predictionNet as predict
import serverGui as gui
import PySimpleGUI as sg


def detectorWrapper2(detect, serverOB, start, window, q, predFrame):
    #continues adding health indexes after the first 100.
    try:
        counter = 0
        while serverOB.server.connected:
            dataTensor, data = serverOB.dataSample()
            result, HI = detect.predict(dataTensor)
            sinceStart = time.time() - start
            send = (HI, data)
            window.window.write_event_value('-CURRENT_HI-', send)
            rows = [HI, sinceStart, 1, sinceStart, 'PREDFRAME']
            predFrame[len(predFrame) - 51] = rows
            for i in range(1, 51):
                futureTime = sinceStart + i * 2.4
                rows = [0, futureTime, 1, futureTime, 'PREDFRAME']
                if i < 50:
                    predFrame[len(predFrame) - 50 + i] = rows
                else:
                    predFrame = np.vstack((predFrame, rows))
            window.window.write_event_value('-PREDICTIONS-', predFrame)
            counter += 1
    except server.ClientStop:
        print("Client Stopped?")
        q.put(1)
    finally:
        return

def detectorWrapper(detect, serverOB, start, window, q):
    #builds the first 100 health indexes.
    try:
        predFrame = np.array([[50, 0, 0, 0, 'PREDFRAME']], dtype=object)
        if len(predFrame) <= 100:
            for counter in range(99):
                dataTensor, data = serverOB.dataSample()
                result, HI = detect.predict(dataTensor)
                sinceStart = time.time() - start
                send = (HI, data)
                window.window.write_event_value('-CURRENT_HI-', send)
                rows = [HI, sinceStart, 1, sinceStart, 'PREDFRAME']
                if counter == 0:
                    predFrame[0] = rows
                predFrame = np.vstack((predFrame, rows))
            for i in range(1, 51):
                futureTime = sinceStart + i*2.4
                rows = [0, futureTime, 1, futureTime, 'PREDFRAME']
                predFrame = np.vstack((predFrame, rows))
            window.window.write_event_value('-PREDICTIONS-', predFrame)
            detectorWrapper2(detect, serverOB, start, window, q, predFrame)
    except server.ClientStop:
        q.put(1)
    return


def connect(serverOB, window):
    connected = serverOB.connect()
    if not connected:
        window.window["-STATUS-"].update("Trying Again.")
        window.window.write_event_value('-UNCONNECTED-', 0)
    else:
        window.window.write_event_value('-CONNECTED-', 1)
    return


def main():
    loading = gui.loadingPopup()
    myServer = server.server('192.168.2.13')
    loading.window['PROGBAR'].update_bar(100/(5-1))
    model = detector.detectorNet()
    loading.window['PROGBAR'].update_bar(100/(5-2))
    predModel = predict.predictNet()
    loading.window['PROGBAR'].update_bar(100/(5-3))
    queueObj = Queue.Queue()
    loading.window['PROGBAR'].update_bar(100/(5-4))
    loading.window.close()
    mainGui = gui.gui()
    threadConnect = threading.Thread(
        target=connect, args=(myServer, mainGui), daemon=True)
    threadConnect.start()
    try:
        while not myServer.server.connected:
            mainGui.window["-STATUS-"].update(
                "Server initialization complete. Ready to recieve data.")
            events, values = mainGui.window.read(timeout=10)
            if events == sg.WIN_CLOSED or events == 'Exit':
                break
            if events == '-UNCONNECTED-' and not threadConnect.is_alive():
                threadConnect.run()
            if events == '-CONNECTED-':
                mainGui.window["-STATUS-"].update("Connected!")
        if myServer.server.connected:
            start = time.time()
            detectorThread = threading.Thread(target=detectorWrapper, args=(
                model, myServer, start, mainGui, queueObj), daemon=True)
            detectorThread.start()
        while myServer.server.connected:
            if not queueObj.empty():
                exc = queueObj.get(block=False)
                if exc == 1:
                    raise server.ClientStop
            events, values = mainGui.window.read(timeout=10)
            if events in (sg.WIN_CLOSED, 'Exit'):
                break
            if events == '-CURRENT_HI-':
                elapsed = time.time() - start
                HI, data = values[events]
                mainGui.HIDraw(HI, elapsed)
                mainGui.sensorDraw(data[1], data[0], data[2], elapsed)
            if events == '-PREDICTIONS-':
                predFrame = values[events]
                dataPred = pd.DataFrame(predFrame, columns=[
                    'health_index', 'time_from_start', 'categorical_id', 't', 'id'], index=list(range(len(predFrame))))
                p10, p50, p90 = predModel.predict(dataPred)
                #dataPred.to_csv("predIn.csv")
                # p10.to_csv("p10.csv")
                # p50.to_csv("p50.csv")
                # p90.to_csv("p90.csv")
                #print(p10)
                #print(p50)
                #print(p90)
                p10np = p10.to_numpy()[0, 2:]
                p50np = p50.to_numpy()[0, 2:]
                p90np = p90.to_numpy()[0, 2:]
                mainGui.forecastDraw(p10np, p50np, p90np, dataPred['time_from_start'][len(predFrame) - 50:])
    except Queue.Empty:
        pass
    except KeyboardInterrupt:
        print("User terminated the server!")
    except server.ClientStop:
        print("Client Disconnected! Terminating")
    finally:
        myServer.terminate()
    return


if __name__ == '__main__':
    main()

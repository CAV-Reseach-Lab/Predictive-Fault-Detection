import PySimpleGUI as sg
import numpy as np
import matplotlib as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

BORDERCOLOR = '#C7D5E0'
FONT='Any 20'
THEME = 'Material 1'
class gui():
    def __init__(self):
        sg.theme(THEME)
        plt.use("TkAgg")
        PADTOP = ((20,20),(20,20))
        PADBANNER = ((800,20),(10,20))
        PADSTATUS = ((550,20),(10,20))
        PADSTATUSCOLUMN = ((20,20),(10,10))
        PADBOTTOM = ((20,20),(10,10))
        PADRIGHT = ((20,10),(0,10))
        CANVASSIZE = (9.45,4.5)
        self.TIMETOPREDICT = 240
        
        self.dataCurrentHI = []
        self.p10 = []
        self.p50 = []
        self.p90 = []
        self.AccX = []
        self.AccY = []
        self.AccZ = []
        self.GyroX = []
        self.GyroY = []
        self.GyroZ = []
        self.speed = []
        self.timestamp = []
        self.timestampForecast = []
        self.timestampSensor = []
        self.xlim = [0,self.TIMETOPREDICT]
        self.ylimAcc = [-10,10]
        self.ylim = [-60,60]

        self.topBanner = [[sg.Text("Detector Net Dashboard",size=(20,1), justification='center', font=FONT, pad=PADBANNER,background_color=BORDERCOLOR)]]
        self.statusBlock = [[sg.Text(size=(50,1), key="-STATUS-", font=FONT, background_color=BORDERCOLOR, pad=PADSTATUS, justification='center')]]
        self.HIBlock = [[sg.Text("HI Graph",justification='center', font=FONT, background_color=BORDERCOLOR)],
                        [sg.Canvas(key="-HICURRENT-")]]
        self.SensorBlock = [[sg.Text("Sensors",justification='center', font=FONT, background_color=BORDERCOLOR)],
                            [sg.Canvas(key="-SENSOR-")]]
        self.layout = [[sg.Column(self.topBanner, size=(1920,50), pad=PADTOP, element_justification='center', background_color=BORDERCOLOR)],
                        [sg.Column(self.statusBlock, size=(1920,50), pad=PADSTATUSCOLUMN,element_justification='center', background_color=BORDERCOLOR)],
                        [sg.Column(self.SensorBlock,size=(960,500), pad=PADBOTTOM,element_justification='center', background_color=BORDERCOLOR),
                        sg.Column(self.HIBlock, size=(960,500), pad=PADBOTTOM, background_color=BORDERCOLOR)]
                        ]
        self.window = sg.Window("Detector Net Dashboard", layout=self.layout, margins=(0,0), grab_anywhere=True, finalize=True)
        
        self.HIcanvas = self.window["-HICURRENT-"].TKCanvas
        self.sensorCanvas = self.window["-SENSOR-"].TKCanvas

        HIfig = plt.figure.Figure(figsize=CANVASSIZE)
        
        SensorFig = plt.figure.Figure(figsize=CANVASSIZE)

        self.currentHI = HIfig.add_subplot(111)
        
        self.currentHI.set_xlim(self.xlim)
        self.currentHI.set_ylim(self.ylim)
        self.currentHI.set_xlabel("Time (s)")
        self.currentHI.set_ylabel("Health Index")
        self.currentHI.grid()
        self.currentHiAgg = draw_figure(self.HIcanvas, HIfig)

        #axs = SensorFig.subplots(3,1, sharex='all')
        #self.sensorsAcc, self.sensorsGyro ,self.sensorsEnc = axs.flatten()
        self.sensorsAcc = SensorFig.add_subplot(311, sharex=self.currentHI)
        self.sensorsGyro = SensorFig.add_subplot(312, sharex=self.currentHI)
        self.sensorsEnc = SensorFig.add_subplot(313, sharex=self.currentHI)

        self.sensorsAcc.set_xlim(self.xlim)
        self.sensorsGyro.set_xlim(self.xlim)
        self.sensorsEnc.set_xlim(self.xlim)

        self.sensorsAcc.set_ylim(self.ylimAcc)

        self.sensorsAcc.set_xlabel("Time (s)")
        self.sensorsAcc.set_ylabel("m/s/s")
        self.sensorsGyro.set_xlabel("Time (s)")
        self.sensorsGyro.set_ylabel("degrees")
        self.sensorsEnc.set_xlabel("Time (s)")
        self.sensorsEnc.set_ylabel("m/s")

        self.sensorsAcc.grid()
        self.sensorsEnc.grid()
        self.sensorsGyro.grid()

        self.sensorsAcc.title.set_text("Accelerometer")
        self.sensorsGyro.title.set_text("Gyroscope")
        self.sensorsEnc.title.set_text("Encoder (Linear Speed)")

        SensorFig.tight_layout()

        self.sensorAgg = draw_figure(self.sensorCanvas, SensorFig)
        return

    def drawGraph(self):
        '''draws graph based on the object's graphing data.'''
        self.currentHI.cla()
        self.currentHI.grid()
        self.currentHI.set_xlim(self.xlim)
        self.currentHI.set_ylim(self.ylim)
        self.currentHI.plot(self.timestampForecast, self.p10, color='red', label='p10')
        self.currentHI.plot(self.timestampForecast, self.p50, color='green', label='p50')
        self.currentHI.plot(self.timestampForecast, self.p90, color='yellow', label='p90')
        self.currentHI.plot(self.timestamp, self.dataCurrentHI, color='blue',label='Current HI')
        self.currentHI.legend(loc='lower right')
        self.currentHI.set_xlabel("Time (s)")
        self.currentHI.set_ylabel("Health Index")
        self.currentHiAgg.draw()


    def HIDraw(self, HI, time_from_start):
        self.dataCurrentHI.append(HI)
        self.timestamp.append(time_from_start)
        if len(self.timestamp) % 100 == 0:
            self.xlim = [self.xlim[0]+self.TIMETOPREDICT, self.xlim[1]+self.TIMETOPREDICT]
        self.drawGraph()
    
    
    def forecastDraw(self, p10, p50, p90, time_from_start):
        self.timestampForecast = np.array(time_from_start)
        self.timestampForecast = self.timestampForecast.astype('float64')
        #print(self.timestampForecast)
        #print(p10)
        self.p10 = p10
        self.p50 = p50
        self.p90 = p90
        #for i in range(50):
        #    time = time_from_start + (2.4*i)
        #    self.timestampForecast.append(time)
        #    self.p10.append(p10[i])
        #    self.p50.append(p50[i])
        #    self.p90.append(p90[i])
    
    def sensorDraw(self,acc, gyro, enc, time_from_start):
        self.timestampSensor.append(time_from_start)
        self.AccX.append(acc[0])
        self.AccY.append(acc[1])
        self.AccZ.append(acc[2])
        self.GyroX.append(gyro[0])
        self.GyroY.append(gyro[1])
        self.GyroZ.append(gyro[2])
        self.speed.append(enc)
        #print(enc)

        self.sensorsAcc.cla()
        self.sensorsGyro.cla()
        self.sensorsEnc.cla()

        self.sensorsAcc.grid()
        self.sensorsGyro.grid()
        self.sensorsEnc.grid()

        self.sensorsAcc.set_xlim(self.xlim)
        self.sensorsGyro.set_xlim(self.xlim)
        self.sensorsEnc.set_xlim(self.xlim)

        #self.sensorsAcc.set_ylim(self.ylimAcc)

        self.sensorsAcc.plot(self.timestampSensor, self.AccX, color='red', label='X')
        self.sensorsAcc.plot(self.timestampSensor, self.AccY, color='green', label='Y')
        self.sensorsAcc.plot(self.timestampSensor, self.AccZ, color='blue', label='Z')

        self.sensorsGyro.plot(self.timestampSensor, self.GyroX, color='red', label='X')
        self.sensorsGyro.plot(self.timestampSensor, self.GyroY, color='green', label='Y')
        self.sensorsGyro.plot(self.timestampSensor, self.GyroZ, color='blue', label='Z')

        self.sensorsEnc.plot(self.timestampSensor, self.speed, color='red', label='Speed')

        self.sensorsAcc.legend(loc='lower right')
        self.sensorsGyro.legend(loc='lower right')
        self.sensorsEnc.legend(loc='lower right')

        self.sensorsAcc.set_xlabel("Time (s)")
        self.sensorsAcc.set_ylabel("(m/s/s)")
        self.sensorsGyro.set_xlabel("Time (s)")
        self.sensorsGyro.set_ylabel("(degrees)")
        self.sensorsEnc.set_xlabel("Time (s)")
        self.sensorsEnc.set_ylabel("(m/s)")

        self.sensorsAcc.title.set_text("Accelerometer")
        self.sensorsGyro.title.set_text("Gyroscope")
        self.sensorsEnc.title.set_text("Encoder (Linear Speed)")

        self.sensorAgg.draw()
    
    def __del__(self):
        self.window.close()

class loadingPopup():
    def __init__(self):
        sg.theme(THEME)
        self.layout = [[sg.Text("Loading...", font=FONT)], [sg.ProgressBar(50, orientation='h',size=(20,20), key='PROGBAR', bar_color=('green', 'red'))]]
        self.window = sg.Window("Loading...",self.layout,finalize=True)
    def __del__(self):
        self.window.close()
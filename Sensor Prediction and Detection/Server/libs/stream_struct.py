from quanser.communications import Stream, StreamError, PollFlag, Timeout
#import struct
#import numpy as np

class StructStream:
    '''Class object consisting of basic stream server/client functionality'''
    def __init__(self, uri, agent='s', send_buffer_size=2048, recv_buffer_size=2048):
        """
        This functions simplifies functionality of the quanser_stream module to provide a 
        simple blocking server or client. \n \n

        INPUTS: \n
        uri - IP server and port in one string, eg. 'tcpip://IP_ADDRESS:PORT' \n
        agent - 's' or 'c' string representing server or client respectively
        send_buffer_size - (optional) size of send buffer, default is 2048 \n
        recv_buffer_size - (optional) size of recv buffer, default is 2048 \n

        """
        self.agent = agent
        self.send_buffer_size = send_buffer_size
        self.recv_buffer_size = recv_buffer_size
        self.uri = uri
        
        # If the agent is a Client, then Server isn't needed. 
        # If the agent is a Server, a Client will also be needed. The server can start listening immediately.
        
        self.clientStream = Stream()
        if agent=='s':
            self.serverStream = Stream()
            
        # Set polling timeout to 1 second, and initialize counter for polling connections
        self.t_out = Timeout(seconds=0, nanoseconds=1000000)
        # counter = 0

        # connected flag initialized to False
        self.connected = False
        non_blocking = False

        try:
            if agent == 'c':
                self.connected = self.clientStream.connect(uri, non_blocking, self.send_buffer_size, self.recv_buffer_size)
       
            elif agent == 's':
                self.serverStream.listen(self.uri, non_blocking)

        except StreamError as e:
            if self.agent == 's':
                print('Server initialization failed.')
            elif self.agent == 'c':
                print('Client initialization failed.')
            print(e.get_error_message())

    def checkConnection(self):

        if self.agent == 'c' and not self.connected:
            try:
                poll_result = self.clientStream.poll(self.t_out, PollFlag.CONNECT)
                                
                if (poll_result & PollFlag.CONNECT) == PollFlag.CONNECT:
                    self.connected = True
                    print('Connected to the Server successfully.')            

            except StreamError as e:
                if e.error_code == -33:
                    self.connected = self.clientStream.connect(self.uri, True, self.send_buffer_size, self.recv_buffer_size)
                else:
                    print('Client initialization failed.')
                    print(e.get_error_message())

        if self.agent == 's' and not self.connected:
            try:
                poll_result = self.serverStream.poll(self.t_out, PollFlag.ACCEPT)
                if (poll_result & PollFlag.ACCEPT) == PollFlag.ACCEPT:
                    self.connected = True
                    print('Found a Client successfully.')
                    self.clientStream = self.serverStream.accept(self.send_buffer_size, self.recv_buffer_size)

            except StreamError as e:
                print('Server initialization failed.')
                print(e.get_error_message())

    def terminate(self):
        if self.connected:
            self.clientStream.shutdown()
            self.clientStream.close()
            print('Successfully terminated clients.')

        if self.agent == 's':
            self.serverStream.shutdown()
            self.serverStream.close()
            print('Successfully terminated servers.')

    def receive(self, buffer, iterations=1000):
        """
        This functions receives a struct buffer object that it will fill with bytes if available. \n \n

        INPUTS: \n
        buffer -  byute buffer input \n
        iterations - (optional) number of times to poll for incoming data before terminating, default is 1000 \n 

        OUTPUTS: \n
        buffer - data received in buffer form\n
        bytes_received - number of bytes received \n
        """
        
        self.t_out = Timeout(1)
        counter = 0
        
        # Calculate total number of bytes needed and set up the bytearray to receive that
        totalNumBytes = len(buffer)
        self.data = bytearray(buffer)
        self.bytes_received = 0        

        # Poll to see if data is incoming, and if so, receive it. Poll a max of 'iteration' times
        try:
            while True:

                # See if data is available
                poll_result = self.clientStream.poll(self.t_out, PollFlag.RECEIVE)
                counter += 1
                if not (iterations == 'Inf'):
                    if counter >= iterations:
                        break        
                if not ((poll_result & PollFlag.RECEIVE) == PollFlag.RECEIVE):
                    continue # Data not available, skip receiving

                # Receive data
                self.bytes_received = self.clientStream.receive(self.data, totalNumBytes)
                
                # data received, so break this loop
                break 

            #  convert byte array back into numpy array and reshape.
            buffer = self.data

        except StreamError as e:
            print(e.get_error_message())
            raise e
        finally:
            return buffer, self.bytes_received

    def send(self, buffer):
        """
        This functions sends the data in the numpy array buffer
        (server or client). \n \n

        INPUTS: \n
        buffer - numpy array of data to be sent \n

        OUTPUTS: \n
        bytesSent - number of bytes actually sent (-1 if send failed) \n
        """

        # Set up array to hold bytes to be sent
        byteArray = bytearray(buffer)
        self.bytesSent = 0
        
        # Send bytes and flush immediately after
        try:
            self.bytesSent = self.clientStream.send(byteArray, len(byteArray))
            self.clientStream.flush()
        except StreamError as e:
            print(e.get_error_message())
            self.bytesSent = -1 # If an error occurs, set bytesSent to -1 for user to check
            raise e
        finally:
            return self.bytesSent

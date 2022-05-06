# %%

import time
import logging
from binance.lib.utils import config_logging

from binance.websocket.futures.websocket_client import FuturesWebsocketClient as Client
from multiprocessing import Process

config_logging(logging, logging.DEBUG)

# %%


class RingBuffer(Process):

    def __init__(self, ddata):
        
        super().__init__()
    
        self.data = [] 
        self.ddata = ddata
        self.is_running = True

    def run(self, ddata):
        while self.is_running:
            self.process_stream()
            # print(self.data[-1])
            time.sleep(1)
    
    def process_stream(self):
        
        for event in self.data[-1]:
            self.ddata[event['s']] = event['c']


#%%







# %%
class ProcessStream(Process):

    def __init__(self, ddata):
        
        super().__init__()
    
        self.data = [] 
        self.ddata = ddata
        self.is_running = True

    def run(self, ddata):
        while self.is_running:
            # self.process_stream()
            for event in self.data[-1]:
                ddata[event['s']] = event['c']
            # print(self.data[-1])
            time.sleep(1)
    
    # def process_stream(self):
    # 

ddata = {}
rb = ProcessStream(ddata)
rb.start()        


#%%
def write_data(m, rb):
    rb.data.append(m)
    time.sleep(1)

client = Client()
client.start()



client.ticker(
        id=13,
        callback= lambda m: write_data(m, rb),
        )


#%%

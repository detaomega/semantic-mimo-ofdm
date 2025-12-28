import uhd
import numpy as np
import time
import config_ofdm as config

def transmit():
    usrp = uhd.usrp.MultiUSRP("type=b200")
    usrp.set_tx_rate(config.SAMPLE_RATE)
    usrp.set_tx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_tx_gain(config.TX_GAIN)
    
    # 封包結構: [靜音] + [S&C Preamble (A+A)] + [Data] + [Data]...
    silence = np.zeros(50, dtype=np.complex64)
    # 重複 5 個 Data
    payload = np.tile(config.DATA_SYMBOL, 5)
    packet = np.concatenate([silence, config.SC_PREAMBLE, payload, silence])
    
    streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.TXMetadata()
    
    print("發送 Schmidl & Cox 訊號...")
    while True:
        md.start_of_burst = True; md.end_of_burst = True
        streamer.send(packet, md)
        time.sleep(0.001)

if __name__ == "__main__": transmit()
import uhd
import numpy as np
import time
import argparse
import config_qpsk as config

def transmit(serial=""):
    device_args = "type=b200"
    if serial: device_args += f",serial={serial}"
    
    print(f"--- OFDM-QPSK 發射機 ---")
    print(f"Gain: {config.TX_GAIN} dB")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_tx_rate(config.SAMPLE_RATE)
    usrp.set_tx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_tx_gain(config.TX_GAIN)
    
    # 封包結構：[靜音] + [Preamble] + [Data] + [Data] + [靜音]
    # 連續發送兩個 Data Symbol 以增加穩定性
    silence = np.zeros(50, dtype=np.complex64)
    packet = np.concatenate([
        silence, 
        config.PREAMBLE_SYMBOL, 
        config.DATA_SYMBOL, 
        config.DATA_SYMBOL, 
        silence
    ])
    
    streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.TXMetadata()
    
    print(">> 正在發送 OFDM + QPSK 訊號...")
    
    try:
        while True:
            md.start_of_burst = True
            md.end_of_burst = True
            streamer.send(packet, md)
            time.sleep(0.001) 
            
    except KeyboardInterrupt:
        print("\nStop")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="")
    args = parser.parse_args()
    transmit(args.serial)
import uhd
import numpy as np
import time
import argparse
import config_qpsk as config

def transmit(serial=""):
    device_args = "type=b200"
    if serial: device_args += f",serial={serial}"
    
    print(f"--- QPSK 發射機 ---")
    print(f"Gain: {config.TX_GAIN} dB")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_tx_rate(config.SAMPLE_RATE)
    usrp.set_tx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_tx_gain(config.TX_GAIN)
    
    # 建立封包：重複發送 QPSK 序列
    # 為了效率，我們一次發送重複 10 次的大封包
    tx_data = np.tile(config.QPSK_SEQ, 10)
    
    streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.TXMetadata()
    
    print(">> 正在連續發送 QPSK 星座點...")
    
    try:
        while True:
            md.start_of_burst = True
            md.end_of_burst = True
            streamer.send(tx_data, md)
            # 極短暫停，避免 buffer underflow 但保持連續性
            time.sleep(0.001) 
            
    except KeyboardInterrupt:
        print("\nStop")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="")
    args = parser.parse_args()
    transmit(args.serial)
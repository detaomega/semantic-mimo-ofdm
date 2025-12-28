import uhd
import numpy as np
import time
import argparse
import config

def transmit_beacon(serial=""):
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
    
    print(f"初始化發射機 (TX) | 裝置: {device_args}")
    usrp = uhd.usrp.MultiUSRP(device_args)
    
    usrp.set_tx_rate(config.SAMPLE_RATE)
    usrp.set_tx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_tx_gain(config.TX_GAIN)
    
    time.sleep(1)

    silence = np.zeros(200, dtype=np.complex64)
    tx_packet = np.concatenate([silence, config.KNOWN_PILOTS, silence])
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_tx_stream(st_args)
    md = uhd.types.TXMetadata()
    
    print(f"開始發送 Pilot 信標... (Freq: {config.CENTER_FREQ/1e6} MHz)")
    print("按 Ctrl+C 停止")
    
    try:
        while True:
            md.start_of_burst = True
            md.end_of_burst = True
            streamer.send(tx_packet, md)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n停止發送")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="", help="B210 發射機的 Serial Number")
    args = parser.parse_args()
    transmit_beacon(args.serial)
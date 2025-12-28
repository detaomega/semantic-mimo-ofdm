import uhd
import numpy as np
import time
import argparse
import config_ofdm as config

def transmit_ofdm(serial=""):
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
    
    print(f"--- OFDM 發射機 (TX) ---")
    print(f"FFT: {config.FFT_SIZE}, CP: {config.CP_LEN}")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_tx_rate(config.SAMPLE_RATE)
    usrp.set_tx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_tx_gain(config.TX_GAIN)
    
    time.sleep(1)

    # 封包結構：[靜音] + [OFDM符號] + [OFDM符號] ... + [靜音]
    # 我們連續發送 10 個 OFDM 符號，方便接收端做平均
    silence = np.zeros(50, dtype=np.complex64)
    burst_signal = np.tile(config.TIME_SYMBOL, 10) # 重複 10 次
    tx_packet = np.concatenate([silence, burst_signal, silence])
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_tx_stream(st_args)
    md = uhd.types.TXMetadata()
    
    print(">> 正在發送 OFDM 訊號... (高速連發)")
    
    try:
        while True:
            md.start_of_burst = True
            md.end_of_burst = True
            streamer.send(tx_packet, md)
            time.sleep(0.001) # 極短間隔
            
    except KeyboardInterrupt:
        print("\n停止發送")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="", help="TX Serial")
    args = parser.parse_args()
    transmit_ofdm(args.serial)
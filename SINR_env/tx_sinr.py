import uhd
import numpy as np
import time
import argparse
import config

def transmit_beacon(serial=""):
    # 設定 USRP 參數
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
    
    print(f"--- 初始化發射機 (TX) ---")
    print(f"裝置: {device_args}")
    print(f"增益: {config.TX_GAIN} dB")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_tx_rate(config.SAMPLE_RATE)
    usrp.set_tx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_tx_gain(config.TX_GAIN)
    
    # 等待 LO 鎖定
    time.sleep(1)

    # 建立封包：[靜音] + [Pilot序列] + [靜音]
    silence = np.zeros(100, dtype=np.complex64)
    tx_packet = np.concatenate([silence, config.KNOWN_PILOTS, silence])
    
    # 設定串流
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_tx_stream(st_args)
    md = uhd.types.TXMetadata()
    
    print(">> 正在發送 Pilot 信標... (按 Ctrl+C 停止)")
    
    try:
        while True:
            md.start_of_burst = True
            md.end_of_burst = True
            streamer.send(tx_packet, md)
            time.sleep(0.05) # 每 0.05 秒發送一次
            
    except KeyboardInterrupt:
        print("\n停止發送")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="", help="TX Serial Number")
    args = parser.parse_args()
    transmit_beacon(args.serial)
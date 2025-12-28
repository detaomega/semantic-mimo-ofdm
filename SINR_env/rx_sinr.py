import uhd
import numpy as np
import argparse
import time
import config

def calculate_sinr_with_correction(rx_segment, known_pilots):
    """
    進階 SINR 計算：包含 CFO (頻率偏移) 相位校正
    """
    N = len(known_pilots)
    
    # --- 步驟 1: 相位校正 (De-rotation) ---
    # 計算接收訊號與已知訊號的相位差向量
    # rx = tx * exp(j * theta)  =>  rx * conj(tx) = |tx|^2 * exp(j * theta)
    phase_diffs = rx_segment * np.conj(known_pilots)
    
    # 計算平均相位偏移 (這是我們要把訊號轉回來的角度)
    # 使用 np.sum 向量合成後取角度，比直接平均角度更抗雜訊
    avg_phase_vector = np.sum(phase_diffs)
    avg_phase = np.angle(avg_phase_vector)
    
    # 進行校正：將接收訊號反向旋轉
    rx_corrected = rx_segment * np.exp(-1j * avg_phase)
    
    # --- 步驟 2: MMSE 估計 (使用校正後的訊號) ---
    # 計算相關性 (此時相位已對齊，相關性應該會最大化)
    correlation = np.sum(rx_corrected * np.conj(known_pilots))
    S_hat = (np.abs(correlation) / N) ** 2
    
    # --- 步驟 3: 功率計算 ---
    P_total = np.mean(np.abs(rx_corrected) ** 2)
    W_hat = P_total - S_hat
    
    # --- 步驟 4: 算出 dB ---
    if W_hat <= 1e-10:
        return 99.9 # 極佳訊號
    
    return 10 * np.log10(S_hat / W_hat)

def run_receiver(serial=""):
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
        
    print(f"--- 初始化接收機 (RX - 進階校正版) ---")
    print(f"裝置: {device_args}")
    print(f"增益: {config.RX_GAIN} dB")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    usrp.set_rx_antenna("TX/RX")
    
    # 設定抓取長度
    num_samps_requested = 5000
    recv_buffer = np.zeros((1, num_samps_requested), dtype=np.complex64)
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_rx_stream(st_args)
    md = uhd.types.RXMetadata()
    
    print(">> 開始監測環境 SINR... (顯示強度 > 2.0 的訊號)")
    print("-" * 65)
    print(f"{'強度 (Peak)':<12} | {'SINR (dB)':<10} | {'訊號品質'}")
    print("-" * 65)
    
    try:
        while True:
            # 1. Burst 接收 (避免 Overflow)
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps_requested
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            # 2. 等待數據填滿
            samps_recvd = 0
            while samps_recvd < num_samps_requested:
                num = streamer.recv(recv_buffer[:, samps_recvd:], md)
                if md.error_code != uhd.types.RXMetadataErrorCode.none:
                    break
                samps_recvd += num
            
            # 3. 處理數據
            data = recv_buffer[0, :samps_recvd]
            
            # 同步: 尋找 Pilot
            corr = np.correlate(data, config.KNOWN_PILOTS, mode='valid')
            
            if len(corr) > 0:
                peak_idx = np.argmax(np.abs(corr))
                peak_val = np.abs(corr[peak_idx])
                
                # --- 門檻值設為 2.0 (確保能抓到微弱訊號) ---
                if peak_val > 2.0: 
                    # 提取 Pilot
                    rx_segment = data[peak_idx : peak_idx + config.N_PILOTS]
                    
                    if len(rx_segment) == config.N_PILOTS:
                        # 使用進階校正算法計算 SINR
                        sinr_db = calculate_sinr_with_correction(rx_segment, config.KNOWN_PILOTS)
                        
                        # 視覺化
                        bar_len = int(max(0, sinr_db))
                        print(f"{peak_val:<12.2f} | {sinr_db:<10.2f} | {'#' * bar_len}")
            
            # 4. 休息
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n停止接收")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="", help="RX Serial Number")
    args = parser.parse_args()
    run_receiver(args.serial)
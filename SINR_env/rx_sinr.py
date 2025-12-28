import uhd
import numpy as np
import argparse
import time
import config

def calculate_mmse_sinr(rx_segment, known_pilots):
    """
    實作 MMSE 演算法計算 SINR [cite: 200, 202]
    S_hat = (1/N) * |sum(Y * C*)|^2
    W_hat = P_total - S_hat
    """
    N = len(known_pilots)
    
    # 1. 估計訊號功率 (S_hat)
    correlation = np.sum(rx_segment * np.conj(known_pilots))
    # 注意：這裡除以 N 是為了取平均相關性，然後平方得到功率
    S_hat = (np.abs(correlation) / N) ** 2
    
    # 2. 計算總接收功率 (P_total)
    P_total = np.mean(np.abs(rx_segment) ** 2)
    
    # 3. 估計干擾加雜訊功率 (W_hat)
    W_hat = P_total - S_hat
    
    # 4. 計算 SINR (dB)
    if W_hat <= 1e-10:
        return 99.9 # 訊號極其純淨
    
    return 10 * np.log10(S_hat / W_hat)

def run_receiver(serial=""):
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
        
    print(f"--- 初始化接收機 (RX) ---")
    print(f"裝置: {device_args}")
    print(f"增益: {config.RX_GAIN} dB (已調整為安全數值)")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    usrp.set_rx_antenna("TX/RX")
    
    # 設定每次抓取的長度 (5000點)
    num_samps_requested = 5000
    recv_buffer = np.zeros((1, num_samps_requested), dtype=np.complex64)
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_rx_stream(st_args)
    md = uhd.types.RXMetadata()
    
    print(">> 開始監測環境 SINR... (只顯示強度 > 10 的訊號)")
    print("-" * 60)
    print(f"{'強度 (Peak)':<12} | {'SINR (dB)':<10} | {'訊號品質'}")
    print("-" * 60)
    
    try:
        while True:
            # --- 1. 發送接收指令 (Burst Mode) 避免 Overflow ---
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps_requested
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            # --- 2. 接收數據 ---
            samps_recvd = 0
            while samps_recvd < num_samps_requested:
                num = streamer.recv(recv_buffer[:, samps_recvd:], md)
                if md.error_code != uhd.types.RXMetadataErrorCode.none:
                    break
                samps_recvd += num
            
            # --- 3. 數據處理 ---
            data = recv_buffer[0, :samps_recvd]
            
            # 同步: 尋找 Pilot
            corr = np.correlate(data, config.KNOWN_PILOTS, mode='valid')
            
            if len(corr) > 0:
                peak_idx = np.argmax(np.abs(corr))
                peak_val = np.abs(corr[peak_idx])
                
                # --- 關鍵修正：門檻值設為 10.0 ---
                # 這樣可以過濾掉您之前看到的 3.91, 4.80 等雜訊
                if peak_val > 10.0: 
                    # 提取訊號段
                    rx_segment = data[peak_idx : peak_idx + config.N_PILOTS]
                    
                    if len(rx_segment) == config.N_PILOTS:
                        sinr_db = calculate_mmse_sinr(rx_segment, config.KNOWN_PILOTS)
                        
                        # 視覺化輸出
                        bar_len = int(max(0, sinr_db))
                        print(f"{peak_val:<12.2f} | {sinr_db:<10.2f} | {'#' * bar_len}")
                else:
                    # 訊號太弱或只是雜訊，直接忽略，保持畫面乾淨
                    pass
            
            # --- 4. 休息一下 ---
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n停止接收")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="", help="B210 接收機的 Serial Number")
    args = parser.parse_args()
    run_receiver(args.serial)
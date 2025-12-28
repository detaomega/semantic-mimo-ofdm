import uhd
import numpy as np
import argparse
import config

def calculate_mmse_sinr(rx_segment, known_pilots):
    """
    實作文件中的 MMSE 演算法來計算 SINR
    參照文件公式 (2.1) 及實作章節 [cite: 200, 202, 211]
    """
    N = len(known_pilots)
    
    # 1. 估計訊號功率 (S_hat)
    # 利用接收訊號與已知 Pilot 的互相關 (Correlation)
    correlation = np.sum(rx_segment * np.conj(known_pilots))
    S_hat = (np.abs(correlation) / N) ** 2
    
    # 2. 計算總接收功率 (P_total)
    P_total = np.mean(np.abs(rx_segment) ** 2)
    
    # 3. 估計干擾加雜訊功率 (W_hat)
    # 總功率 = 訊號功率 + (干擾+雜訊)功率
    W_hat = P_total - S_hat
    
    # 4. 計算 SINR (dB)
    if W_hat <= 1e-10:
        return 99.9 # 避免除以零 (代表訊號極其純淨)
    
    return 10 * np.log10(S_hat / W_hat)

def run_receiver(serial=""):
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
        
    print(f"初始化接收機 (RX) | 裝置: {device_args}")
    usrp = uhd.usrp.MultiUSRP(device_args)
    
    # B210 接收設定
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    usrp.set_rx_antenna("TX/RX") # 確保天線接在 TX/RX 孔
    
    # 設定 Buffer (需大於 Tx 的封包長度以確保能捕捉完整訊號)
    buffer_len = 2000 
    recv_buffer = np.zeros((1, buffer_len), dtype=np.complex64)
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_rx_stream(st_args)
    md = uhd.types.RXMetadata()
    
    # 開始連續接收
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)
    
    print("正在監測環境 SINR... (按 Ctrl+C 停止)")
    print("-" * 50)
    
    try:
        while True:
            # 接收數據
            num_samps = streamer.recv(recv_buffer, md)
            
            # 錯誤處理 (例如 Overflow)
            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
                
            data = recv_buffer[0, :num_samps]
            
            # --- 步驟 A: 同步 (Synchronization) ---
            # 使用 Cross-Correlation 尋找 Pilot 在這個 Buffer 中的位置
            corr = np.correlate(data, config.KNOWN_PILOTS, mode='valid')
            peak_idx = np.argmax(np.abs(corr))
            peak_val = np.abs(corr[peak_idx])
            
            # 設定門檻值 (Threshold)，避免在沒有訊號時計算雜訊
            # 這個值可能需要根據環境調整，如果一直沒反應請調低
            if peak_val > 5: 
                
                # --- 步驟 B: 提取訊號 ---
                # 取得 Pilot 開始到結束的那一段數據
                rx_segment = data[peak_idx : peak_idx + config.N_PILOTS]
                
                # 確保長度正確 (避免切到 Buffer 邊緣)
                if len(rx_segment) == config.N_PILOTS:
                    
                    # --- 步驟 C: 計算 SINR ---
                    sinr_db = calculate_mmse_sinr(rx_segment, config.KNOWN_PILOTS)
                    
                    # --- 顯示結果 ---
                    # 簡單的視覺化長條圖
                    bar_len = int(max(0, sinr_db))
                    bar = "#" * bar_len
                    print(f"Sync! 強度:{peak_val:.1f} | SINR: {sinr_db:6.2f} dB | {bar}")
            
    except KeyboardInterrupt:
        print("\n停止接收")
    finally:
        # 停止指令
        usrp.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="", help="B210 接收機的 Serial Number")
    args = parser.parse_args()
    run_receiver(args.serial)
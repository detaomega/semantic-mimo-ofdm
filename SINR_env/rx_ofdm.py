import uhd
import numpy as np
import argparse
import time
import config_ofdm as config

def calculate_ofdm_sinr(rx_symbols):
    """
    OFDM SINR 計算器
    rx_symbols: 接收到的多個 OFDM 符號 (頻域)，形狀為 (N_symbols, 52)
    """
    # 1. 通道估測 (Channel Estimation) - LS 方法
    # H = Rx / Tx
    # 取平均以消除雜訊影響
    rx_avg = np.mean(rx_symbols, axis=0)
    H_est = rx_avg / config.qpsk_symbols
    
    # 2. 等化 (Equalization)
    # Rx_eq = Rx / H_est
    rx_equalized = rx_symbols / H_est
    
    # 3. 計算誤差 (EVM Noise)
    # 誤差 = 等化後的訊號 - 理想的 QPSK 訊號
    error_vector = rx_equalized - config.qpsk_symbols
    
    # 4. 計算功率
    # 訊號功率 (理想)
    signal_power = np.mean(np.abs(config.qpsk_symbols) ** 2)
    # 雜訊功率 (誤差的變異數)
    noise_power = np.mean(np.abs(error_vector) ** 2)
    
    # 5. SINR
    if noise_power <= 1e-12:
        return 99.9
    
    return 10 * np.log10(signal_power / noise_power)

def run_ofdm_receiver(serial=""):
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
        
    print(f"--- OFDM 接收機 (RX) ---")
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    usrp.set_rx_antenna("TX/RX")
    
    # 抓長一點，因為我们要找連續的 OFDM 符號
    num_samps = 10000
    recv_buffer = np.zeros((1, num_samps), dtype=np.complex64)
    
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.RXMetadata()
    
    print(">> 開始監測 OFDM SINR...")
    print(f"{'強度':<10} | {'SINR (dB)':<10} | {'品質'}")
    print("-" * 40)
    
    try:
        while True:
            # --- 修正部分開始 ---
            # 舊寫法 (報錯): streamer.issue_stream_cmd(uhd.types.StreamCMD(..., ..., ...))
            # 新寫法 (修正): 分開設定
            
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            # --- 修正部分結束 ---
            
            samps_recvd = 0
            while samps_recvd < num_samps:
                num = streamer.recv(recv_buffer[:, samps_recvd:], md)
                if md.error_code != uhd.types.RXMetadataErrorCode.none: break
                samps_recvd += num
            
            data = recv_buffer[0, :samps_recvd]
            
            # 2. 同步 (使用 Time Domain Correlation)
            corr = np.correlate(data, config.TIME_SYMBOL, mode='valid')
            peak_idx = np.argmax(np.abs(corr))
            peak_val = np.abs(corr[peak_idx])
            
            if peak_val > 5.0:
                # 3. 提取多個 OFDM 符號
                # 我們試著連續提取 5 個符號來做分析
                collected_symbols = []
                
                # 從第一個 peak 開始抓
                current_idx = peak_idx
                
                for _ in range(5): # 抓 5 個符號
                    # 檢查邊界
                    if current_idx + config.TOTAL_LEN > len(data):
                        break
                        
                    # 切割: [CP + Data]
                    symbol_with_cp = data[current_idx : current_idx + config.TOTAL_LEN]
                    
                    # 去除 CP (Remove Cyclic Prefix) -> 剩下 FFT_SIZE
                    symbol_no_cp = symbol_with_cp[config.CP_LEN:]
                    
                    # FFT 轉頻域
                    freq_domain = np.fft.fft(symbol_no_cp) / np.sqrt(config.FFT_SIZE)
                    
                    # 提取有效子載波 (只看我們有送資料的那 52 個)
                    indices = (config.OCCUPIED_BINS + config.FFT_SIZE) % config.FFT_SIZE
                    active_carriers = freq_domain[indices]
                    
                    collected_symbols.append(active_carriers)
                    
                    # 移動到下一個符號 (跳過 CP + Data)
                    current_idx += config.TOTAL_LEN
                
                # 4. 如果收集足夠，計算 SINR
                if len(collected_symbols) >= 2:
                    rx_stack = np.array(collected_symbols)
                    sinr = calculate_ofdm_sinr(rx_stack)
                    
                    if sinr > 0:
                        bar = "#" * int(sinr)
                        print(f"{peak_val:<10.2f} | {sinr:<10.2f} | {bar}")
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stop")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="")
    args = parser.parse_args()
    run_ofdm_receiver(args.serial)
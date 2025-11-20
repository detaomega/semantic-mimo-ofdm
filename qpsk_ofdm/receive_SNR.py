# receive.py
import numpy as np
import uhd
import matplotlib.pyplot as plt
import time

# --- 設定區 ---
SERIAL_RX = "34B733A"   # 請確認你的 B200mini 序列號
RX_GAIN = 60.0          # 接收增益 (太高可能導致 SNR 誤判)
Fc = 5.2e9
Fs = 1e6

# --- OFDM 參數 (需與 Tx 一致) ---
FFT_size = 64
CP_len = 16
symbol_len = FFT_size + CP_len
num_data_symbols = 10
num_data_carriers = 48
DC_NULLS = 8

# --- 計算索引 (用於 SNR) ---
num_data_carriers_per_side = num_data_carriers // 2
total_used_block = num_data_carriers + DC_NULLS
data_block_start = (FFT_size - total_used_block) // 2 # 4

# 1. 訊號索引 (Data Carriers)
idx_data_left = np.arange(data_block_start, data_block_start + num_data_carriers_per_side)
idx_data_right = np.arange(data_block_start + num_data_carriers_per_side + DC_NULLS, 
                           data_block_start + total_used_block)
idx_signal = np.hstack([idx_data_left, idx_data_right])

# 2. 雜訊索引 (Null Carriers / Guard Bands)
# 使用最外側的子載波作為雜訊參考 (索引 0-3 和 60-63)
idx_null_left = np.arange(0, data_block_start)
idx_null_right = np.arange(data_block_start + total_used_block, FFT_size)
idx_noise = np.hstack([idx_null_left, idx_null_right])

# --- 輔助函式: Schmidl-Cox 同步 ---
def find_frame_start(buffer, D=FFT_size//2):
    # 簡化版 S&C
    if len(buffer) < 4*D: return -1, 0
    
    # 能量計算
    half_len = len(buffer) // 2
    P = np.zeros(half_len, dtype=np.complex64)
    R = np.zeros(half_len, dtype=np.float32)
    
    # 這裡使用簡化的滑動視窗計算 (為了效能，Python loop 較慢，實務上可用 numpy 優化)
    # 這裡直接用簡單的 vector operation
    delay_sig = buffer[D:]
    orig_sig = buffer[:-D]
    n = min(len(delay_sig), len(orig_sig))
    
    # 相關性
    conj_prod = buffer[D:D+n] * np.conj(buffer[:n])
    window = np.ones(D)
    P_metric = np.convolve(conj_prod, window, mode='valid')
    
    # 能量
    pwr_sig = np.abs(buffer[:n])**2
    R_metric = np.convolve(pwr_sig, window, mode='valid')
    
    # 避免除以0
    R_metric[R_metric < 1e-9] = 1e-9
    
    M = (np.abs(P_metric)**2) / (R_metric**2)
    
    peak_idx = np.argmax(M)
    # 如果峰值太低，視為未偵測到
    if M[peak_idx] < 0.3: # 門檻值
        return -1, 0
        
    # 計算 CFO
    angle = np.angle(P_metric[peak_idx])
    cfo = angle / D
    
    return peak_idx, cfo

# --- 主程式 ---
def main():
    # 連接 USRP
    print(f"Connecting to RX USRP ({SERIAL_RX})...")
    usrp_rx = uhd.usrp.MultiUSRP(f"serial={SERIAL_RX}")
    usrp_rx.set_rx_rate(Fs)
    usrp_rx.set_rx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
    usrp_rx.set_rx_gain(RX_GAIN, 0)
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    rx_streamer = usrp_rx.get_rx_stream(st_args)
    metadata = uhd.types.RXMetadata()
    
    # 繪圖初始化
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Environmental SNR Monitor")
    ax.set_xlabel("Frame Count")
    ax.set_ylabel("SNR (dB)")
    snr_history = []
    line, = ax.plot([], [])
    ax.grid(True)
    
    print("\n*** Starting Reception... ***")
    try:
        frame_count = 0
        while True:
            # 1. 接收訊號 Burst
            num_samps = 2000 # 收夠多以包含一個 Frame (約960點)
            buffer = np.zeros(num_samps, dtype=np.complex64)
            
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            cmd.num_samps = num_samps
            cmd.stream_now = True
            rx_streamer.issue_stream_cmd(cmd)
            rx_streamer.recv(buffer, metadata)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
            
            # 2. 同步
            peak, cfo = find_frame_start(buffer)
            
            if peak > 0 and peak + symbol_len*(num_data_symbols+2) < len(buffer):
                # 3. CFO 校正
                t = np.arange(len(buffer))
                buffer_corrected = buffer * np.exp(-1j * cfo * t)
                
                # 4. 提取並處理 Data Symbols (跳過 Preamble 和 LTF)
                # Frame 結構: [Preamble] [LTF] [Data1] [Data2] ...
                # Preamble = 2 symbols (S&C window size effect), LTF = 1 symbol
                # 這裡簡化：從 Peak 開始，往後跳過 Preamble 區段
                # 注意：find_frame_start 的 peak 通常在 preamble 尾端附近
                
                # 定位第一個 Data Symbol 的開頭
                # 假設 peak 是第一個 preamble 的重複點
                # 簡單起見，我們只抓一個 data symbol 來算 SNR 就很準了
                # 或是遍歷所有 data symbols
                
                total_sig_pwr = 0
                total_noise_pwr = 0
                
                # 我們從 peak 往後抓取一段穩定的數據區 (避開 preamble)
                # 假設 Preamble+LTF 約佔 3 個 symbol 長度
                start_idx = peak + symbol_len * 3 
                
                for i in range(num_data_symbols):
                    sym_start = start_idx + i*symbol_len
                    if sym_start + symbol_len > len(buffer): break
                    
                    # 取出 Time Domain Symbol (去掉 CP)
                    sym_time = buffer_corrected[sym_start+CP_len : sym_start+symbol_len]
                    
                    # FFT
                    sym_freq = np.fft.fft(sym_time)
                    
                    # --- 關鍵: SNR 計算 (Physical) ---
                    # 訊號能量 = Data Carriers 的平均能量
                    p_sig = np.mean(np.abs(sym_freq[idx_signal])**2)
                    
                    # 雜訊能量 = Null Carriers (Guard Band) 的平均能量
                    p_noise = np.mean(np.abs(sym_freq[idx_noise])**2)
                    
                    total_sig_pwr += p_sig
                    total_noise_pwr += p_noise
                
                # 計算平均 SNR
                if total_noise_pwr > 0:
                    avg_snr_linear = total_sig_pwr / total_noise_pwr
                    avg_snr_db = 10 * np.log10(avg_snr_linear)
                    
                    snr_history.append(avg_snr_db)
                    if len(snr_history) > 50: snr_history.pop(0)
                    
                    print(f"\rSync! Frame: {frame_count} | Env SNR: {avg_snr_db:.2f} dB | CFO: {cfo:.4f}", end="")
                    
                    # 更新繪圖
                    line.set_data(range(len(snr_history)), snr_history)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
                    frame_count += 1
            else:
                pass
                # print(".", end="") # 沒訊號時印點點

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plt.close()

if __name__ == "__main__":
    main()
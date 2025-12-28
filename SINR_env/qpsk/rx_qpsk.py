import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import config_qpsk as config

def calculate_evm_sinr(rx_symbols):
    """
    計算 Error Vector Magnitude (EVM) 並轉為 SINR
    rx_symbols: 已修正通道效應的接收點
    """
    # 1. 硬判決 (Hard Decision) - 找最近的理想 QPSK 點
    # 理想點是 1+j, -1+j, -1-j, 1-j (歸一化後)
    # 簡單作法：判斷象限
    signs = np.sign(rx_symbols.real) + 1j * np.sign(rx_symbols.imag)
    ideal_points = signs / np.sqrt(2) # 歸一化到單位圓
    
    # 2. 計算誤差向量
    error_vector = rx_symbols - ideal_points
    
    # 3. 計算功率
    signal_power = np.mean(np.abs(ideal_points)**2) # 應為 1
    noise_power = np.mean(np.abs(error_vector)**2)
    
    if noise_power < 1e-10: return 50.0 # 保護
    
    sinr = 10 * np.log10(signal_power / noise_power)
    return sinr, rx_symbols

def run_rx(serial=""):
    device_args = "type=b200"
    if serial: device_args += f",serial={serial}"
    
    print(f"--- OFDM-QPSK 接收分析儀 ---")
    print(f"RX Gain: {config.RX_GAIN} dB")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    
    # 準備繪圖
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    line_const, = ax.plot([], [], 'b.', alpha=0.5, markersize=3, label='Rx Data')
    # 畫理想點
    ax.plot([0.707, -0.707, -0.707, 0.707], [0.707, 0.707, -0.707, -0.707], 'r+', markersize=12, label='Ideal')
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.grid(True); ax.legend()
    ax.set_title("OFDM-QPSK Constellation (Equalized)")
    text_info = ax.text(-1.8, 1.7, "Waiting...", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # 接收緩衝
    num_samps = 5000
    buff = np.zeros((1, num_samps), dtype=np.complex64)
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.RXMetadata()
    
    # S&C 參數
    L = config.FFT_SIZE // 2
    
    try:
        while True:
            # 1. 接收 (三行式設定)
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            samps_recvd = 0
            while samps_recvd < num_samps:
                num = streamer.recv(buff[:, samps_recvd:], md)
                samps_recvd += num
            data = buff[0, :samps_recvd]
            
            # 2. S&C 同步
            # 延遲相關
            delay_prod = data[:-L] * np.conj(data[L:])
            window = np.ones(L)
            P = np.convolve(delay_prod, window, mode='valid')
            energy = np.abs(data[L:])**2
            R = np.convolve(energy, window, mode='valid')
            
            # Metric
            with np.errstate(divide='ignore', invalid='ignore'):
                M = (np.abs(P)**2) / (R[:len(P)]**2 + 1e-10)
            
            peak_idx = np.argmax(M)
            
            # 門檻判定
            if M[peak_idx] > 0.6:
                # 3. 頻偏估計 (CFO)
                angle = np.angle(P[peak_idx])
                cfo_est = angle / (2 * np.pi) * (config.SAMPLE_RATE / L)
                
                # 4. 提取 Preamble 和 Data
                # Preamble 結束點 (也是 Data CP 開始點)
                data_start_idx = peak_idx + config.FFT_SIZE # 因為 Preamble 是 A+A (2L = FFT_SIZE)
                
                if data_start_idx + config.TOTAL_LEN <= len(data):
                    
                    # 抓取原始數據 (包含 Preamble 和 Data)
                    # 為了做好的 CFO 修正，我們從 Preamble 開始抓
                    raw_chunk = data[peak_idx : data_start_idx + config.TOTAL_LEN]
                    
                    # 建立全域時間軸進行 CFO 修正
                    t = np.arange(len(raw_chunk))
                    correction = np.exp(-1j * 2 * np.pi * cfo_est * t / config.SAMPLE_RATE)
                    chunk_corrected = raw_chunk * correction
                    
                    # 切割
                    preamble_corrected = chunk_corrected[:config.FFT_SIZE]
                    data_corrected = chunk_corrected[config.FFT_SIZE:] # 這是 [CP + Data]
                    
                    # --- 頻域處理 ---
                    
                    # A. 通道估測 (Channel Estimation)
                    # Preamble FFT
                    rx_pre_freq = np.fft.fft(preamble_corrected) / np.sqrt(config.FFT_SIZE)
                    # H = Rx / Tx (只使用偶數載波，因為奇數是 0)
                    # 簡單起見，我們只用偶數點算 H，然後插值
                    H_est = np.zeros(config.FFT_SIZE, dtype=np.complex64)
                    
                    valid_indices = np.abs(config.KNOWN_PREAMBLE_FREQ) > 0.1
                    H_est[valid_indices] = rx_pre_freq[valid_indices] / config.KNOWN_PREAMBLE_FREQ[valid_indices]
                    
                    # 對奇數點做插值 (填補 0 的部分)
                    for i in range(1, config.FFT_SIZE, 2):
                        prev_h = H_est[i-1]
                        next_h = H_est[(i+1)%config.FFT_SIZE]
                        H_est[i] = (prev_h + next_h) / 2
                    
                    # 平滑化 H (去除雜訊)
                    H_est = np.convolve(H_est, np.ones(3)/3, mode='same')
                    
                    # B. Data 解調
                    # 去 CP
                    data_no_cp = data_corrected[config.CP_LEN:]
                    rx_data_freq = np.fft.fft(data_no_cp) / np.sqrt(config.FFT_SIZE)
                    
                    # 等化 (Equalization)
                    # 避免除以 0
                    H_safe = H_est.copy()
                    H_safe[np.abs(H_safe) < 1e-3] = 1e-3
                    rx_eq = rx_data_freq / H_safe
                    
                    # 提取有效子載波 (Data Carriers)
                    indices = (config.OCCUPIED_BINS + config.FFT_SIZE) % config.FFT_SIZE
                    rx_payload = rx_eq[indices]
                    
                    # 5. 計算 SINR
                    sinr, rx_final_points = calculate_evm_sinr(rx_payload)
                    
                    # 繪圖更新
                    line_const.set_data(np.real(rx_final_points), np.imag(rx_final_points))
                    
                    text_info.set_text(
                        f"Metric: {M[peak_idx]:.2f}\n"
                        f"CFO: {cfo_est:.0f} Hz\n"
                        f"SINR: {sinr:.2f} dB"
                    )
                    
                    if sinr > 15:
                        text_info.set_color("green")
                    else:
                        text_info.set_color("red")
                        
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stop")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="")
    args = parser.parse_args()
    run_rx(args.serial)
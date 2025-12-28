import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import config_qpsk as config

def calculate_evm_sinr(rx_symbols):
    """ 計算 SINR """
    # 硬判決
    signs = np.sign(rx_symbols.real) + 1j * np.sign(rx_symbols.imag)
    ideal_points = signs / np.sqrt(2)
    
    # 誤差與功率
    error_vector = rx_symbols - ideal_points
    signal_power = np.mean(np.abs(ideal_points)**2)
    noise_power = np.mean(np.abs(error_vector)**2)
    
    if noise_power < 1e-10: return 50.0, rx_symbols
    
    sinr = 10 * np.log10(signal_power / noise_power)
    return sinr, rx_symbols

def run_rx(serial=""):
    device_args = "type=b200"
    if serial: device_args += f",serial={serial}"
    
    print(f"--- OFDM-QPSK 接收機 (Pilot 相位修正版) ---")
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    
    # 繪圖
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    line_const, = ax.plot([], [], 'b.', alpha=0.5, markersize=3, label='Rx Data')
    ax.plot([0.707, -0.707, -0.707, 0.707], [0.707, 0.707, -0.707, -0.707], 'r+', markersize=12, label='Ideal')
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.grid(True); ax.legend()
    ax.set_title("Corrected Constellation")
    text_info = ax.text(-1.8, 1.7, "Waiting...", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    num_samps = 5000
    buff = np.zeros((1, num_samps), dtype=np.complex64)
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.RXMetadata()
    
    L = config.FFT_SIZE // 2
    
    try:
        while True:
            # 1. 接收 (三行式)
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
            delay_prod = data[:-L] * np.conj(data[L:])
            window = np.ones(L)
            P = np.convolve(delay_prod, window, mode='valid')
            R = np.convolve(np.abs(data[L:])**2, window, mode='valid')
            
            with np.errstate(divide='ignore', invalid='ignore'):
                M = (np.abs(P)**2) / (R[:len(P)]**2 + 1e-10)
            
            peak_idx = np.argmax(M)
            
            if M[peak_idx] > 0.6:
                # 3. 粗略頻偏修正 (Coarse CFO)
                cfo_est = np.angle(P[peak_idx]) / (2 * np.pi) * (config.SAMPLE_RATE / L)
                
                # 4. 提取 Preamble + Data
                data_start_idx = peak_idx + config.FFT_SIZE
                if data_start_idx + config.TOTAL_LEN <= len(data):
                    
                    raw_chunk = data[peak_idx : data_start_idx + config.TOTAL_LEN]
                    
                    # 應用粗略 CFO 修正
                    t = np.arange(len(raw_chunk))
                    correction = np.exp(-1j * 2 * np.pi * cfo_est * t / config.SAMPLE_RATE)
                    chunk_corrected = raw_chunk * correction
                    
                    # 切割
                    preamble_corrected = chunk_corrected[:config.FFT_SIZE]
                    data_corrected = chunk_corrected[config.FFT_SIZE:]
                    
                    # --- 頻域處理 ---
                    
                    # A. 通道估測 (H)
                    rx_pre_freq = np.fft.fft(preamble_corrected) / np.sqrt(config.FFT_SIZE)
                    H_est = np.zeros(config.FFT_SIZE, dtype=np.complex64)
                    valid_idx = np.abs(config.KNOWN_PREAMBLE_FREQ) > 0.1
                    H_est[valid_idx] = rx_pre_freq[valid_idx] / config.KNOWN_PREAMBLE_FREQ[valid_idx]
                    
                    # 插值補滿奇數點
                    for i in range(1, config.FFT_SIZE, 2):
                        H_est[i] = (H_est[i-1] + H_est[(i+1)%config.FFT_SIZE]) / 2
                    
                    H_est = np.convolve(H_est, np.ones(3)/3, mode='same') # 平滑
                    
                    # B. Data 解調與等化
                    data_no_cp = data_corrected[config.CP_LEN:]
                    rx_data_freq = np.fft.fft(data_no_cp) / np.sqrt(config.FFT_SIZE)
                    
                    H_safe = H_est.copy()
                    H_safe[np.abs(H_safe) < 1e-3] = 1e-3
                    rx_eq = rx_data_freq / H_safe
                    
                    # --- 關鍵步驟：利用 Pilot 進行殘餘相位修正 (Residual Phase Tracking) ---
                    
                    # 1. 抓出接收到的 Pilot (在等化後)
                    pilot_indices = (config.PILOT_BINS + config.FFT_SIZE) % config.FFT_SIZE
                    rx_pilots = rx_eq[pilot_indices]
                    
                    # 2. 計算它跟理想 Pilot (1+0j) 的相位差
                    # 因為我們發送的是 1.0，所以 rx_pilots 應該要是 1.0
                    # 任何偏差都是相位旋轉
                    pilot_phase_diff = np.angle(rx_pilots)
                    
                    # 3. 平均這個相位差 (Common Phase Error)
                    avg_phase_error = np.mean(pilot_phase_diff)
                    
                    # 4. 將整個符號 (包含 Data) 轉回來
                    phase_correction_factor = np.exp(-1j * avg_phase_error)
                    rx_eq_phased = rx_eq * phase_correction_factor
                    
                    # --- 結束修正 ---
                    
                    # C. 提取 Data 算 SINR
                    data_indices = (config.DATA_BINS + config.FFT_SIZE) % config.FFT_SIZE
                    rx_payload = rx_eq_phased[data_indices]
                    
                    sinr, _ = calculate_evm_sinr(rx_payload)
                    
                    line_const.set_data(np.real(rx_payload), np.imag(rx_payload))
                    text_info.set_text(f"M: {M[peak_idx]:.2f}\nSINR: {sinr:.2f} dB")
                    text_info.set_color("green" if sinr > 15 else "red")
                    
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
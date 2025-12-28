import uhd
import numpy as np
import time
import config_ofdm as config

def run_rx():
    usrp = uhd.usrp.MultiUSRP("type=b200")
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    
    # 這裡的 L 是半個 Preamble 的長度
    L = len(config.SC_PREAMBLE) // 2 
    
    num_samps = 5000
    buff = np.zeros((1, num_samps), dtype=np.complex64)
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.RXMetadata()
    
    print("S&C 接收機啟動... (Auto-Correlation)")
    print(f"{'Metric':<10} | {'CFO (Hz)':<10} | {'SINR (dB)':<10}")
    print("-" * 40)
    
    while True:
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = num_samps; stream_cmd.stream_now = True
        streamer.issue_stream_cmd(stream_cmd)
        streamer.recv(buff, md)
        data = buff[0]
        
        # --- Schmidl & Cox 算法核心 ---
        # 計算 P(d) = sum(r[d+m] * r*[d+m+L])
        # 這是在比較「這一段」跟「L點之後的那一段」像不像
        
        # 為了效能，我們簡化實作：只對每個點做延遲共軛相乘
        # delay_corr = data[i] * conj(data[i+L])
        delay_prod = data[:-L] * np.conj(data[L:])
        
        # 滑動視窗總和 (Moving Sum) - 這裡用簡單的卷積模擬
        window = np.ones(L)
        P = np.convolve(delay_prod, window, mode='valid')
        
        # 能量 R (Normalization)
        energy = np.abs(data)**2
        R = np.convolve(energy, window, mode='valid')
        R = R[:len(P)] # 對齊長度
        
        # S&C Metric: M = |P|^2 / R^2
        with np.errstate(divide='ignore', invalid='ignore'):
            M = (np.abs(P)**2) / (R**2 + 1e-10)
            
        # 找最大值
        peak_idx = np.argmax(M)
        peak_val = M[peak_idx]
        
        # S&C 的 Peak 理論最大值是 1.0 (完全吻合)
        # 通常 > 0.6 就是很好的訊號
        if peak_val > 0.4: 
            # 1. 計算 CFO
            # angle(P) 就是相位差
            phase_diff = np.angle(P[peak_idx])
            # CFO = angle / (2 * pi * T_L), T_L = L / Fs
            cfo_est = phase_diff / (2 * np.pi) * (config.SAMPLE_RATE / L)
            
            # 2. 提取 Data 算 SINR (加上 CFO 修正)
            # Preamble 結束後就是 Data
            data_start = peak_idx + len(config.SC_PREAMBLE)
            if data_start + config.FFT_SIZE + config.CP_LEN < len(data):
                # 取出 Data (包含 CP)
                raw_sym = data[data_start : data_start + config.FFT_SIZE + config.CP_LEN]
                
                # 修正 CFO
                t = np.arange(len(raw_sym))
                # 注意頻率修正公式
                correction = np.exp(-1j * 2 * np.pi * cfo_est * t / config.SAMPLE_RATE)
                sym_corrected = raw_sym * correction
                
                # 去 CP, FFT
                sym_no_cp = sym_corrected[config.CP_LEN:]
                rx_freq = np.fft.fft(sym_no_cp) / np.sqrt(config.FFT_SIZE)
                
                # 簡單 EVM SINR
                # H_est (Blind) = rx / known
                H = rx_freq / config.KNOWN_QPSK
                rx_eq = rx_freq / H
                noise = np.mean(np.abs(rx_eq - config.KNOWN_QPSK)**2)
                sinr = 10*np.log10(1/noise) if noise > 0 else 99
                
                print(f"{peak_val:<10.2f} | {cfo_est:<10.0f} | {sinr:<10.2f}")

        time.sleep(0.1)

if __name__ == "__main__": run_rx()
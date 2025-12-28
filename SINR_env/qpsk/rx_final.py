import numpy as np
import uhd
import matplotlib.pyplot as plt
import argparse
import time

# --- 參數設定 ---
# 請修改您的 RX 序號
SERIAL_RX = "34B733A"   
RX_GAIN = 60.0          
Fc = 5.4e9              
Fs = 1e6                

# --- OFDM 參數 (需與 Tx 一致) ---
FFT_size = 64
CP_len = 16
symbol_len = FFT_size + CP_len 
N_SAMPLES_PER_FRAME = 960 
N_DATA_SYMBOLS = 10

# Pilot 與 Data 索引
PILOT_CARRIERS = np.array([-21, -7, 7, 21])
ALL_CARRIERS = np.arange(-26, 27)
ALL_CARRIERS = ALL_CARRIERS[ALL_CARRIERS != 0]
DATA_CARRIERS = np.setdiff1d(ALL_CARRIERS, PILOT_CARRIERS)

pilot_indices = (PILOT_CARRIERS + FFT_size) % FFT_size
data_indices = (DATA_CARRIERS + FFT_size) % FFT_size

# --- 核心函式: 環境 SINR 計算 (EVM) ---
def calculate_environmental_sinr(rx_payload):
    """
    Blind EVM SINR 計算
    """
    # 1. 歸一化
    avg_pwr = np.mean(np.abs(rx_payload))
    if avg_pwr == 0: return -99.0
    rx_norm = rx_payload / avg_pwr
    
    # 2. 硬判決 (找最近的理想 QPSK 點)
    # QPSK 理想點歸一化後位於 45, 135, 225, 315 度
    signs = np.sign(np.real(rx_norm)) + 1j * np.sign(np.imag(rx_norm))
    ideal_points = signs * (np.mean(np.abs(rx_norm))) # 映射回原振幅
    
    # 3. 計算誤差
    error_vector = rx_norm - ideal_points
    signal_power = np.mean(np.abs(ideal_points)**2)
    noise_power = np.mean(np.abs(error_vector)**2)
    
    # 4. 轉 dB
    if noise_power < 1e-10: return 50.0
    return 10 * np.log10(signal_power / noise_power)

# --- S&C 同步 ---
def find_schmidl_cox_peak(buffer, D):
    search_len = len(buffer) // 2
    if search_len < 2 * D: return -1, 0

    window_A = buffer[:search_len-D]
    window_B = buffer[D:search_len]
    P = np.convolve(window_B * window_A.conj(), np.ones(D), 'valid')
    R_A = np.convolve(np.abs(window_A)**2, np.ones(D), 'valid')
    R_B = np.convolve(np.abs(window_B)**2, np.ones(D), 'valid')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        M = (np.abs(P)**2) / (R_A * R_B + 1e-10)
    
    peak_idx = np.argmax(M)
    cfo_phase = np.angle(P[peak_idx]) / D
    
    # 門檻
    if M[peak_idx] < 0.6: return -1, 0 
    return peak_idx - CP_len, cfo_phase

def run_rx(serial):
    # 載入 LTF 驗證檔
    try:
        ltf_data = np.load('ltf_data.npz')
        ltf_known = ltf_data['ltf_freq_known'] 
    except:
        print("錯誤: 找不到 ltf_data.npz，請先執行 Tx 程式")
        return

    print(f"--- OFDM 接收機 (SINR分析版) ---")
    dev_addr = f"serial={serial}" if serial else "type=b200"
    usrp_rx = uhd.usrp.MultiUSRP(dev_addr)
    usrp_rx.set_rx_rate(Fs)
    usrp_rx.set_rx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
    usrp_rx.set_rx_gain(RX_GAIN, 0)
    
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [0]
    rx_streamer = usrp_rx.get_rx_stream(stream_args)
    metadata = uhd.types.RXMetadata()
    
    # 繪圖初始化
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    
    print("開始接收... (按 Ctrl+C 停止)")
    
    try:
        while True:
            # 1. 接收 Burst
            num_samps = int(Fs * 0.2) # 0.2秒
            buff = np.zeros(num_samps, dtype=np.complex64)
            
            # 使用穩定的三行式指令
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            cmd.num_samps = num_samps
            cmd.stream_now = True
            rx_streamer.issue_stream_cmd(cmd)
            
            samps_recvd = 0
            while samps_recvd < num_samps:
                num = rx_streamer.recv(buff[samps_recvd:], metadata)
                if num == 0: break
                samps_recvd += num
            
            if samps_recvd < N_SAMPLES_PER_FRAME: continue
            
            # 2. 同步
            peak_idx, cfo_phase = find_schmidl_cox_peak(buff, FFT_size // 2)
            if peak_idx < 0: 
                print(".", end="", flush=True)
                continue
            
            # 3. CFO 校正
            corr_vec = np.exp(-1j * cfo_phase * np.arange(len(buff)))
            buff_corrected = buff * corr_vec
            frame = buff_corrected[peak_idx : peak_idx + N_SAMPLES_PER_FRAME]
            
            if len(frame) < N_SAMPLES_PER_FRAME: continue

            # 4. 通道估測 (LTF)
            ltf_sym = frame[symbol_len : symbol_len*2]
            ltf_freq = np.fft.fft(ltf_sym[CP_len:])
            H = np.zeros_like(ltf_freq)
            valid = np.abs(ltf_known) > 0.1
            H[valid] = ltf_freq[valid] / ltf_known[valid]
            # 簡單插值
            for i in range(1, 64): 
                if H[i] == 0: H[i] = H[i-1]

            # 5. 解調與相位追蹤
            rx_constellation = []
            
            for i in range(N_DATA_SYMBOLS):
                sym_start = (i + 2) * symbol_len
                sym = frame[sym_start : sym_start + symbol_len]
                sym_freq = np.fft.fft(sym[CP_len:])
                
                # A. 等化
                sym_eq = sym_freq / (H + 1e-10)
                
                # B. Pilot 相位追蹤 (消除旋轉)
                rx_pilots = sym_eq[pilot_indices]
                avg_phase_err = np.mean(np.angle(rx_pilots)) # 理想是 0度
                sym_eq_tracked = sym_eq * np.exp(-1j * avg_phase_err)
                
                # C. 取出數據
                rx_data = sym_eq_tracked[data_indices]
                rx_constellation.append(rx_data)
                
            rx_payload = np.concatenate(rx_constellation)
            
            # 6. 計算 SINR
            env_sinr = calculate_environmental_sinr(rx_payload)
            
            print(f"\nSYNC! CFO:{cfo_phase:.4f} | Env SINR: {env_sinr:.2f} dB")
            
            # 7. 繪圖
            ax1.clear()
            ax1.plot(np.abs(H), '.-')
            ax1.set_title("Channel Magnitude")
            ax1.grid(True)
            
            ax2.clear()
            ax2.scatter(np.real(rx_payload), np.imag(rx_payload), s=5, alpha=0.6)
            ax2.scatter([0.707, -0.707], [0.707, -0.707], c='r', marker='+', s=100) # 僅示範理想點
            ax2.scatter([0.707, -0.707], [-0.707, 0.707], c='r', marker='+', s=100)
            ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2); ax2.grid(True)
            ax2.set_title(f"Constellation (SINR={env_sinr:.1f}dB)")
            
            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default=SERIAL_RX)
    args = parser.parse_args()
    run_rx(args.serial)
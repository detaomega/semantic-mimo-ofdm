import numpy as np
import uhd
import matplotlib.pyplot as plt
import time

# --- 設定 ---
SERIAL_RX = "34B733A"   # 您的 B200mini 序列號
RX_GAIN = 60.0          # 接收增益 (dB)
Fc = 5.1e9              # 載波頻率 (必須與 Tx 相同)
Fs = 1e6                # 取樣率 (1 MHz)

# --- OFDM 參數 (必須與 Tx 相同) ---
FFT_size = 64
CP_len = 16
symbol_len = FFT_size + CP_len # 80
N_SAMPLES_PER_FRAME = 960 # 12 symbols * 80 samples/symbol
N_DATA_SYMBOLS = 10
N_DATA_CARRIERS = 48
data_carrier_start = (FFT_size - N_DATA_CARRIERS) // 2
data_carrier_end = data_carrier_start + N_DATA_CARRIERS


def find_schmidl_cox_peak(buffer, D, L):
    """
    執行 Schmidl-Cox 同步演算法 (自相關)
    D = 
    L = 
    """
    # P[n] = sum( r[n+k+D] * conj(r[n+k]) ) for k=0..D-1
    # R[n] = sum( |r[n+k+D]|^2 ) for k=0..D-1
    # M[n] = |P[n]|^2 / (R[n] * R[n]) # (R[n]**2 會不穩定)
    
    # 我們只搜尋緩衝區的前半部分，以節省時間
    search_len = len(buffer) // 2
    
    # 這是 P[n] 的滑動總和 (Correlation)
    P_n = buffer[D:search_len] * buffer[:search_len-D].conj()
    P = np.convolve(P_n, np.ones(D), 'valid')

    # 這是 R[n] 的滑動總和 (Power)
    R_n = np.abs(buffer[D:search_len])**2 + np.abs(buffer[:search_len-D])**2
    R = np.convolve(R_n, np.ones(D), 'valid')
    
    # 避免除以零
    R[R == 0] = 1e-10
    
    M = (np.abs(P)**2) / (R**2)
    
    # 找到相關峰值 (這是 S&C 前導碼的開頭)
    peak_idx = np.argmax(M)
    
    # (關鍵) 估計 CFO
    # CFO = angle(P[peak_idx]) / (pi * D) # (正規化頻率)
    cfo_phase_per_sample = np.angle(P[peak_idx]) / D
    
    print(f"Schmidl-Cox: peak at {peak_idx}, CFO phase: {cfo_phase_per_sample:.4f} rad/sample")
    
    return peak_idx, cfo_phase_per_sample

# --- 1. 讀取 Ground Truth 檔案 (由 Tx 產生) ---
print("Loading Ground Truth and LTF files...")
try:
    gt_data = np.load('gt_simple.npz')
    gt_payload_freq = gt_data['gt_payload_freq'] # 10x48
    
    ltf_data = np.load('ltf_data.npz')
    ltf_freq_known = ltf_data['ltf_freq_known'] # 1x64
except FileNotFoundError:
    print("!!! 錯誤: 找不到 'gt_simple.npz' 或 'ltf_data.npz' !!!")
    print("請先執行 tx_ofdm_simple.py 來產生這些檔案。")
    exit()

# --- 2. 連接並設定 B200mini ---
print(f"Connecting to RX USRP (B200mini) at serial={SERIAL_RX}...")
usrp_rx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_RX}"))
usrp_rx.set_clock_source("internal", 0)
usrp_rx.set_time_source("internal", 0)
usrp_rx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
usrp_rx.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A"), 0)
usrp_rx.set_rx_antenna("RX2", 0)
usrp_rx.set_rx_rate(Fs)
usrp_rx.set_rx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_rx.set_rx_gain(RX_GAIN, 0)
print(f"B200mini (Rx) setup complete. Rate: {Fs/1e6} MHz, Freq: {Fc/1e9} GHz, Gain: {RX_GAIN} dB")

# --- 3. 建立串流 ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
rx_streamer = uhd.usrp.get_rx_stream(stream_args)

# --- 4. 離線接收 (關鍵修改) ---
# 準備一個大緩衝區 (3 秒)
num_samps_to_recv = int(Fs * 3.0) 
recv_buffer = np.zeros(num_samps_to_recv, dtype=np.complex64)
metadata = uhd.types.RXMetadata()

# 準備繪圖
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))
fig.tight_layout(pad=4.0)

try:
    # 3. 開始接收
    print(f"*** Starting OFFLINE reception... (Receiving {num_samps_to_recv} samples) ***")
    print("請在此時啟動 tx_ofdm_simple.py")
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_samps_and_done)
    stream_cmd.num_samps = num_samps_to_recv
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    
    # 4. 接收數據 (這會阻塞，直到 3 秒的數據都收到)
    total_rx_samps = 0
    while total_rx_samps < num_samps_to_recv:
        samps = rx_streamer.recv(recv_buffer[total_rx_samps:], metadata)
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f"Receiver Error: {metadata.strerror()}")
        total_rx_samps += samps
        
    print(f"Reception complete. Total samples: {total_rx_samps}")
    
    # 5. 離線同步 (S&C)
    print("Starting offline synchronization (Schmidl-Cox)...")
    peak_idx, cfo_phase = find_schmidl_cox_peak(recv_buffer, FFT_size // 2, FFT_size)
    
    if peak_idx <= 0 or (peak_idx + N_SAMPLES_PER_FRAME) > len(recv_buffer):
        print(f"!!! SYNC FAILED !!! (peak_idx: {peak_idx})")
        print("無法在訊號中找到 S&C 前導碼。")
        ax1.psd(recv_buffer, NFFT=1024, Fs=Fs)
        ax1.set_title("PSD (Sync Failed)")
        plt.show(block=True)
        exit()

    print(f"*** SYNC SUCCESS! *** Found frame at index: {peak_idx}")

    # 6. (關鍵) CFO 校正
    # 建立一個與緩衝區一樣長的校正向量
    cfo_corr_vector = np.exp(-1j * cfo_phase * np.arange(len(recv_buffer)))
    recv_buffer_corrected = recv_buffer * cfo_corr_vector
    
    print("CFO correction applied.")

    # 7. 擷取訊框 (從校正後的緩衝區)
    frame = recv_buffer_corrected[peak_idx : peak_idx + N_SAMPLES_PER_FRAME]
    
    # 8. 通道估測
    # 擷取 LTF 符號 (符號 1)
    ltf_symbol = frame[symbol_len : symbol_len*2]
    ltf_symbol_no_cp = ltf_symbol[CP_len:]
    ltf_freq_rx = np.fft.fft(ltf_symbol_no_cp)
    
    # 計算通道 H = Y/X
    channel_H = ltf_freq_rx / ltf_freq_known
    
    # 9. 解碼數據
    all_rx_payloads = []
    
    for i in range(N_DATA_SYMBOLS):
        symbol_start = (i + 2) * symbol_len # +2 是因為 0=SC, 1=LTF
        symbol_end = symbol_start + symbol_len
        
        data_symbol = frame[symbol_start : symbol_end]
        data_symbol_no_cp = data_symbol[CP_len:]
        
        data_freq_rx = np.fft.fft(data_symbol_no_cp)
        
        # 均衡 Y_eq = Y / H
        data_freq_equalized = data_freq_rx / channel_H
        
        # 擷取數據子載波
        payload = data_freq_equalized[data_carrier_start:data_carrier_end]
        all_rx_payloads.append(payload)
        
    rx_payload_freq = np.vstack(all_rx_payloads)

    # 10. 計算 ESNR (選用，但很棒)
    sym_pow = np.mean(np.abs(gt_payload_freq)**2)
    err_pow = np.mean(np.abs(gt_payload_freq - rx_payload_freq)**2)
    esnr = 10*np.log10(sym_pow / err_pow)
    
    print(f"\n--- FINAL RESULT ---")
    print(f"ESNR: {esnr:.2f} dB")
    
    # 11. 繪製最終結果
    print("Plotting final result...")
    
    # 圖 1: 通道
    ax1.clear()
    ax1.plot(np.abs(channel_H[data_carrier_start:data_carrier_end]), '.-')
    ax1.set_title(f'Channel Magnitude (H) | ESNR: {esnr:.2f} dB')
    ax1.set_xlabel('Data Subcarrier Index')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True)
    
    # 圖 2: 星座圖
    ax2.clear()
    ax2.scatter(np.real(rx_payload_freq), np.imag(rx_payload_freq), 
                s=1, alpha=0.3, label='Received (Rx)')
    ax2.set_xlim([-1.5, 1.5]) 
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_xlabel('In-Phase')
    ax2.set_ylabel('Quadrature-Phase')
    ax2.set_title('QPSK Constellation')
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    plt.show(block=True) # 保持繪圖視窗開啟

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    print("Receiver shut down.")
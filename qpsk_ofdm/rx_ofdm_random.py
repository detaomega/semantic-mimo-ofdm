import numpy as np
import uhd
import matplotlib.pyplot as plt
import time

# --- 設定 ---
SERIAL_RX = "34B733A"   # 您的 B200mini 序列號
RX_GAIN = 60.0
Fc = 5.4e9
Fs = 1e6

# --- 檔案名稱 (固定) ---
GT_FILE_LTF = 'ltf_data.npz'
GT_FILE_PAYLOAD = 'gt_simple.npz'

# --- OFDM 參數 (必須與 Tx 相同) ---
FFT_size = 64
CP_len = 16
symbol_len = FFT_size + CP_len # 80
N_SAMPLES_PER_FRAME = 960 
N_DATA_SYMBOLS = 10
N_DATA_CARRIERS = 48
data_carrier_start = (FFT_size - N_DATA_CARRIERS) // 2
data_carrier_end = data_carrier_start + N_DATA_CARRIERS

# -----------------------------------------------------------------
# (!!! 關鍵：使用「正確且穩健」的 S&C 同步函式 !!!)
# -----------------------------------------------------------------
def find_schmidl_cox_peak(buffer, D):
    """
    執行 Schmidl-Cox 同步演算法 (自相關)
    D = FFT_size / 2
    """
    search_len = len(buffer) // 2
    if search_len < 2 * D:
        return -1, 0

    window_A = buffer[:search_len-D]
    window_B = buffer[D:search_len]

    P_n = window_B * window_A.conj()
    P = np.convolve(P_n, np.ones(D), 'valid')

    R_A_n = np.abs(window_A)**2
    R_A = np.convolve(R_A_n, np.ones(D), 'valid')

    R_B_n = np.abs(window_B)**2
    R_B = np.convolve(R_B_n, np.ones(D), 'valid')
    
    R_A[R_A == 0] = 1e-10
    R_B[R_B == 0] = 1e-10
    
    M = (np.abs(P)**2) / (R_A * R_B)
    
    peak_idx = np.argmax(M)
    
    cfo_phase_per_sample = np.angle(P[peak_idx]) / D
    
    print(f"Schmidl-Cox: peak at {peak_idx}, CFO phase: {cfo_phase_per_sample:.4f} rad/sample")
    
    return peak_idx - CP_len, cfo_phase_per_sample
# -----------------------------------------------------------------

# --- 1. 讀取 Ground Truth 檔案 (由 Tx 產生) ---
print("Loading Ground Truth and LTF files...")
try:
    gt_data = np.load(GT_FILE_PAYLOAD)
    gt_payload_freq = gt_data['gt_payload_freq'] # 10x48
    
    ltf_data = np.load(GT_FILE_LTF)
    ltf_freq_known = ltf_data['ltf_freq_known'] # 1x64
except FileNotFoundError:
    print(f"!!! 錯誤: 找不到 '{GT_FILE_PAYLOAD}' 或 '{GT_FILE_LTF}' !!!")
    print("請先執行 1_generate_truth_files.py 來產生這些檔案。")
    exit()

# --- 2. 連接並設定 B200mini ---
print(f"Connecting to RX USRP (B200mini) at serial={SERIAL_RX}...")
usrp_rx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_RX}"))
usrp_rx.set_clock_source("internal")
usrp_rx.set_time_source("internal")
usrp_rx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
usrp_rx.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A"), 0)
usrp_rx.set_rx_antenna("RX2", 0) # (請確認你的 Rx 天線埠)
usrp_rx.set_rx_rate(Fs)
usrp_rx.set_rx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_rx.set_rx_gain(RX_GAIN, 0)
print(f"B200mini (Rx) setup complete. Rate: {Fs/1e6} MHz, Freq: {Fc/1e9} GHz, Gain: {RX_GAIN} dB")

# --- 3. 建立串流 ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
rx_streamer = usrp_rx.get_rx_stream(stream_args)
metadata = uhd.types.RXMetadata()

# --- 4. 準備繪圖 ---
print("Setting up live plot...")
plt.ion() # 開啟互動模式
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(18, 5))
fig.tight_layout(pad=4.0)

# --- 5. 主接收迴圈 (離線/突發模式) ---
print("\n*** Starting burst reception loop... (Press Ctrl+C to stop) ***")
try:
    while True:
        # --- A. 接收一個數據突發 (Burst) ---
        num_samps_to_recv = int(Fs * 0.5) # 接收 0.5 秒
        recv_buffer = np.zeros(num_samps_to_recv, dtype=np.complex64)
        temp_buffer = np.zeros(10000, dtype=np.complex64)
        
        stream_cmd_start = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd_start.stream_now = True
        rx_streamer.issue_stream_cmd(stream_cmd_start)
        
        total_rx_samps = 0
        print(f"Receiving {num_samps_to_recv} samples burst...")
        while total_rx_samps < num_samps_to_recv:
            samps = rx_streamer.recv(temp_buffer, metadata)
            if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                print("O", end="", flush=True) 
                continue
            elif metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"Rx Error: {metadata.strerror()}")
            
            samps_to_copy = min(samps, num_samps_to_recv - total_rx_samps)
            if samps_to_copy <= 0:
                break
            
            recv_buffer[total_rx_samps : total_rx_samps + samps_to_copy] = temp_buffer[:samps_to_copy]
            total_rx_samps += samps_to_copy
        
        stream_cmd_stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(stream_cmd_stop)
        
        print("\nBurst reception complete. Processing...")
        rcv_waveform_full = recv_buffer

        # --- B. 離線處理 ---
        
        # 5. 離線同步 (S&C)
        print("Starting offline synchronization (Schmidl-Cox)...")
        peak_idx, cfo_phase = find_schmidl_cox_peak(rcv_waveform_full, FFT_size // 2)
        
        if peak_idx <= 0 or (peak_idx + N_SAMPLES_PER_FRAME) > len(rcv_waveform_full):
            print(f"!!! SYNC FAILED !!! (peak_idx: {peak_idx})")
            print("無法在訊號中找到 S&C 前導碼。")
            continue

        print(f"*** SYNC SUCCESS! *** Found frame at index: {peak_idx}")

        # 6. (關鍵) CFO 校正
        cfo_corr_vector = np.exp(-1j * cfo_phase * np.arange(len(rcv_waveform_full)))
        recv_buffer_corrected = rcv_waveform_full * cfo_corr_vector
        print("CFO correction applied.")

        # 7. 擷取訊框 (從校正後的緩衝區)
        frame = recv_buffer_corrected[peak_idx : peak_idx + N_SAMPLES_PER_FRAME]
        
        # 8. 通道估測
        ltf_symbol = frame[symbol_len : symbol_len*2]
        ltf_symbol_no_cp = ltf_symbol[CP_len:]
        ltf_freq_rx = np.fft.fft(ltf_symbol_no_cp)
        channel_H = ltf_freq_rx / ltf_freq_known
        
        # 9. 解碼數據
        all_rx_payloads = []
        for i in range(N_DATA_SYMBOLS):
            symbol_start = (i + 2) * symbol_len
            symbol_end = symbol_start + symbol_len
            data_symbol = frame[symbol_start : symbol_end]
            data_symbol_no_cp = data_symbol[CP_len:]
            data_freq_rx = np.fft.fft(data_symbol_no_cp)
            
            data_freq_equalized = data_freq_rx / channel_H
            
            payload = data_freq_equalized[data_carrier_start:data_carrier_end]
            all_rx_payloads.append(payload)
            
        rx_payload_freq = np.vstack(all_rx_payloads) # (10, 48)

        # 10. 計算 ESNR
        sym_pow = np.mean(np.abs(gt_payload_freq)**2)
        err_pow = np.mean(np.abs(gt_payload_freq - rx_payload_freq)**2)
        esnr = 10*np.log10(sym_pow / err_pow)
        
        print(f"\n--- FINAL RESULT ---")
        print(f"--- ESNR: {esnr:.2f} dB ---")
        
        # 11. 繪製最終結果
        
        # 圖 1: 通道
        ax1.clear()
        ax1.plot(np.abs(channel_H[data_carrier_start:data_carrier_end]), '.-')
        ax1.set_title(f'Channel Magnitude (H)')
        ax1.set_xlabel('Data Subcarrier Index')
        ax1.set_ylabel('Magnitude')
        ax1.grid(True)
        ax1.set_ylim(bottom=0)
        
        # 圖 2: PSD
        ax2.clear()
        ax2.psd(rcv_waveform_full, NFFT=1024, Fs=Fs, scale_by_freq=False, linewidth=0.5)
        ax2.set_title('Received PSD')
        ax2.set_xlabel('Frequency (Hz)')

        # 圖 3: 星座圖
        ax3.clear()
        ax3.scatter(np.real(rx_payload_freq), np.imag(rx_payload_freq), 
                    s=1, alpha=0.3, label='Received (Rx)')
        ax3.scatter(np.real(gt_payload_freq), np.imag(gt_payload_freq), 
                    s=2, color='orange', label='Ground Truth (GT)')
        ax3.set_xlim([-1.5, 1.5]) 
        ax3.set_ylim([-1.5, 1.5])
        ax3.set_xlabel('In-Phase')
        ax3.set_ylabel('Quadrature-Phase')
        ax3.set_title(f'Arbitrary Constellation | ESNR: {esnr:.2f} dB')
        ax3.grid(True)
        ax3.set_aspect('equal')
        
        # 刷新 GUI
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream_cmd_stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd_stop)
    plt.ioff()
    plt.close()
    print("Receiver shut down.")
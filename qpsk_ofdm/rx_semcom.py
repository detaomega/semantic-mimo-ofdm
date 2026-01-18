import numpy as np
import uhd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# --- 設定 ---
SERIAL_RX = "34B733A"   # 請確認您的 B200mini 序列號
RX_GAIN = 65.0          
Fc = 5.2e9
Fs = 1e6
GT_FILE_LTF = 'ltf_data.npz'
GT_FILE_PAYLOAD = 'tx_payload.npy'

# 設定儲存資料夾
SAVE_DIR = "rx_saved"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"建立儲存資料夾: {SAVE_DIR}")

# --- OFDM 參數 (必須與 Tx 完全相同) ---
FFT_size = 64
CP_len = 16
symbol_len = FFT_size + CP_len 
num_data_carriers = 48
DC_NULLS = 8
num_data_carriers_per_side = num_data_carriers // 2 

# Tx 端: 512 / 48 = 10.66 -> 需要 11 個 Symbol
N_DATA_SYMBOLS = 11 
N_SAMPLES_PER_FRAME = (1 + 1 + N_DATA_SYMBOLS) * symbol_len 

# 子載波索引計算
total_used_block = num_data_carriers + DC_NULLS 
data_block_start = (FFT_size - total_used_block) // 2 
data_left_start = data_block_start
data_left_end = data_left_start + num_data_carriers_per_side 
data_right_start = data_left_end + DC_NULLS 
data_right_end = data_right_start + num_data_carriers_per_side 

# --- S&C 同步函式 ---
def find_schmidl_cox_peak(buffer, D):
    search_len = len(buffer) // 2
    if search_len < 2 * D: return -1, 0
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
    if np.max(M) < 0.5: return -1, 0
    cfo_phase_per_sample = np.angle(P[peak_idx]) / D
    return peak_idx - CP_len, cfo_phase_per_sample

# --- 1. 讀取 Ground Truth ---
print("Loading files...")
try:
    ltf_data = np.load(GT_FILE_LTF)
    if 'ltf_freq_known' in ltf_data:
        ltf_freq_known = ltf_data['ltf_freq_known']
    else:
        ltf_freq_known = ltf_data['arr_0']

    raw_payload = np.load(GT_FILE_PAYLOAD) 
    gt_flat_complex = raw_payload[:, 0] + 1j * raw_payload[:, 1] 
    
    n_padding = N_DATA_SYMBOLS * num_data_carriers - len(gt_flat_complex) 
    gt_padded = np.pad(gt_flat_complex, (0, n_padding), 'constant')
    gt_payload_freq = gt_padded.reshape(N_DATA_SYMBOLS, num_data_carriers) 
    
    print(f"GT loaded. Shape: {gt_payload_freq.shape}")

except FileNotFoundError:
    print(f"!!! 錯誤: 找不到檔案 !!!")
    exit()

# --- 2. 連接 USRP ---
print(f"Connecting to RX USRP...")
usrp_rx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_RX}"))
usrp_rx.set_clock_source("internal")
usrp_rx.set_rx_rate(Fs)
usrp_rx.set_rx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_rx.set_rx_gain(RX_GAIN, 0)
print(f"Rx Setup: {Fc/1e9}GHz, Gain={RX_GAIN}dB")

# --- 3. 串流 ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
rx_streamer = usrp_rx.get_rx_stream(stream_args)
metadata = uhd.types.RXMetadata()

# --- 4. 繪圖設定 ---
plt.ion() 
fig, (ax2, ax3) = plt.subplots(ncols=2, nrows=1, figsize=(14, 6))
fig.tight_layout(pad=3.0)

# --- 5. 接收迴圈 ---
print("\n*** Starting reception... (Press Ctrl+C to stop) ***")
try:
    while True:
        num_samps_to_recv = int(N_SAMPLES_PER_FRAME * 2.5) 
        recv_buffer = np.zeros(num_samps_to_recv, dtype=np.complex64)
        
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = num_samps_to_recv
        stream_cmd.stream_now = True
        rx_streamer.issue_stream_cmd(stream_cmd)
        
        samps = rx_streamer.recv(recv_buffer, metadata)
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none: continue 
        
        rcv_waveform = recv_buffer[:samps]

        peak_idx, cfo_phase = find_schmidl_cox_peak(rcv_waveform, FFT_size // 2)
        if peak_idx < 0 or (peak_idx + N_SAMPLES_PER_FRAME) > len(rcv_waveform): continue 

        cfo_corr = np.exp(-1j * cfo_phase * np.arange(len(rcv_waveform)))
        waveform_corrected = rcv_waveform * cfo_corr
        frame = waveform_corrected[peak_idx : peak_idx + N_SAMPLES_PER_FRAME]
        
        ltf_idx = symbol_len 
        ltf_rx = frame[ltf_idx : ltf_idx + symbol_len]
        ltf_rx_freq = np.fft.fft(ltf_rx[CP_len:]) 
        channel_H = ltf_rx_freq / ltf_freq_known 
        
        all_rx_payloads = []
        for i in range(N_DATA_SYMBOLS):
            start = (i + 2) * symbol_len 
            sym_time = frame[start : start + symbol_len]
            sym_freq = np.fft.fft(sym_time[CP_len:])
            sym_eq = sym_freq / channel_H
            payload = np.hstack([
                sym_eq[data_left_start:data_left_end],
                sym_eq[data_right_start:data_right_end]
            ])
            all_rx_payloads.append(payload)
            
        rx_payload_matrix = np.vstack(all_rx_payloads) 
        
        # --- 計算誤差 ---
        error_vector = rx_payload_matrix - gt_payload_freq
        signal_power = np.mean(np.abs(gt_payload_freq) ** 2)
        noise_power = np.mean(np.abs(error_vector) ** 2)
        if noise_power == 0: noise_power = 1e-12 # 避免除以零
        global_sinr_db = 10 * np.log10(signal_power / noise_power)

        err_pow = np.mean(np.abs(gt_payload_freq - rx_payload_matrix)**2)
        esnr = 10*np.log10(1.0 / err_pow) 

        # --- 儲存邏輯修改 ---
        rx_flat = rx_payload_matrix.flatten() 
        rx_data_final = rx_flat[:512] # 去除 Padding
        
        save_payload = np.stack([np.real(rx_data_final), np.imag(rx_data_final)], axis=1)
        
        # (!!! 關鍵修改: 產生帶時間戳的檔名 !!!)
        timestamp = datetime.now().strftime("%H%M%S_%f")[:9] # 時分秒_微秒(取前3位)
        filename = f"rx_{timestamp}_SINR{global_sinr_db:.4f}.npy"
        file_path = os.path.join(SAVE_DIR, filename)
        
        np.save(file_path, save_payload)
        
        print(f"Saved: {filename} | SINR: {global_sinr_db:.2f} dB")

        # --- 繪圖 ---
        ax2.clear()
        ax2.psd(frame, NFFT=256, Fs=Fs)
        ax2.set_title(f'PSD ({timestamp})')
        
        ax3.clear()
        ax3.scatter(np.real(rx_payload_matrix), np.imag(rx_payload_matrix), 
                    s=5, alpha=0.5, label='Rx')
        ax3.scatter(np.real(gt_payload_freq), np.imag(gt_payload_freq), 
                    s=5, c='orange', alpha=0.5, label='GT')
        ax3.set_title(f'Constellation (MSE: {err_pow:.4f})')
        ax3.set_xlim([-2, 2])
        ax3.set_ylim([-2, 2])
        ax3.grid()
        ax3.legend()
        
        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    plt.close()
    print("Done.")

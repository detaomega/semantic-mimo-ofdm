import numpy as np
import uhd
import time
import matplotlib.pyplot as plt
from scipy.signal import convolve
from ofdm_parameters import * # 導入所有共用參數

# --- 1. 載入發送的 Bits (用於 BER) ---
try:
    tx_bits = np.load('tx_bits.npy')
    print(f"Loaded {len(tx_bits)} tx_bits from tx_bits.npy")
except FileNotFoundError:
    print("Error: tx_bits.npy not found. \n請先在另一個終端機執行 tx_ofdm.py 來產生此檔案。")
    exit()

# --- 2. 輔助函數 (Ported from MATLAB) ---

def sync(rx_data, sts_with_cp):
    print("Synchronizing...")
    sts_matched_filter = np.conj(sts_with_cp[::-1]) # 共軛 + 反轉
    filtered_signal = convolve(rx_data, sts_matched_filter, mode='same')
    
    abs_filtered = np.abs(filtered_signal)
    max_index = np.argmax(abs_filtered)
    max_value = abs_filtered[max_index]
    
    # 簡易門檻值，您可能需要根據實際訊號調整
    noise_floor = np.mean(abs_filtered[:10000]) # 假設開頭是雜訊
    threshold = noise_floor * 50 # 門檻設為雜訊的 50 倍
    
    print(f"Sync peak: {max_value:.2f} at index {max_index} (Threshold: {threshold:.2f})")
    
    # MATLAB 邏輯: 檢查峰值是否足夠大且不在邊緣
    if max_value > threshold and max_index < (len(rx_data) - FRAME_LEN) and max_index > 130:
        start_index = max_index - len(sts_with_cp) // 2
        # 修正：start_index 是 STS 的中心點，減去 STS_LEN//2 才是 STS 的起點
        # start_index = max_index - len(sts_with_cp) // 2 
        # MATLAB 的 start_index 是 frame 起點 (含 zero padding)
        # max_index 應對應到 STS 的中心 (80)，所以 frame 起點是 max_index - 80 - 50
        frame_start = max_index - (len(sts_with_cp) // 2) - START_ZERO_LEN
        print(f"Sync success! Frame starts at index {frame_start}")
        return frame_start, True # 回傳 frame 的起點
    
    return -1, False

def cfo_estimation(rcvdSignal, frame_start_index, fs):
    # STS 粗估
    sts_start = frame_start_index + START_ZERO_LEN
    rx_first_STS = rcvdSignal[sts_start + 112 : sts_start + 112 + 16]
    rx_second_STS = rcvdSignal[sts_start + 128 : sts_start + 128 + 16]
    
    p_k_s = np.dot(np.conj(rx_first_STS), rx_second_STS)
    phi_estimate_s = np.angle(p_k_s)
    delta_f_sts = phi_estimate_s * fs / (2 * np.pi * 16)
    
    # LTS 精估
    t_sts_1 = np.arange(0, 64)
    t_sts_2 = np.arange(64, 128)
    
    idx_lts1_start = frame_start_index + START_ZERO_LEN + STS_LEN + (LTS_LEN - 2 * FFT_SIZE) // 2 # 50+160+32
    rx_first_LTS = rcvdSignal[idx_lts1_start : idx_lts1_start + FFT_SIZE]
    rx_first_LTS = rx_first_LTS * np.exp((-1j * 2 * np.pi * delta_f_sts * t_sts_1) / fs)
    
    idx_lts2_start = idx_lts1_start + FFT_SIZE
    rx_second_LTS = rcvdSignal[idx_lts2_start : idx_lts2_start + FFT_SIZE]
    rx_second_LTS = rx_second_LTS * np.exp((-1j * 2 * np.pi * delta_f_sts * t_sts_2) / fs)
    
    p_k = np.dot(np.conj(rx_first_LTS), rx_second_LTS)
    phi_estimate = np.angle(p_k)
    delta_f = phi_estimate * fs / (2 * np.pi * 64)
    
    delta_f_total = delta_f + delta_f_sts
    print(f"CFO Estimation: STS={delta_f_sts:.2f} Hz, LTS={delta_f:.2f} Hz, Total={delta_f_total:.2f} Hz")
    
    return delta_f_total, rx_first_LTS

def ofdm_output(rx_ofdm):
    num_syms = rx_ofdm.shape[1]
    # 執行 FFT 並正規化
    rx_symbols_before = np.fft.ifftshift(np.fft.fft(rx_ofdm, axis=0), axes=0) / FFT_SIZE
    
    rx_symbols = rx_symbols_before[DATA_INDICES, :]
    Pilot = rx_symbols_before[PILOT_INDICES, :]
    
    return rx_symbols, Pilot

def pilot_tone_correction(rx_symbols, Pilot, delta_f_total, fc):
    k_data = DATA_INDICES - FFT_SIZE // 2
    k_pilot = PILOT_INDICES - FFT_SIZE // 2
    
    epsilon_0 = delta_f_total / fc
    epsilon = epsilon_0
    T_u = (FFT_SIZE + CP_SIZE_DATA) / FS
    
    W = np.zeros(NUM_SYMBOLS, dtype=np.complex64)
    U = 0
    pho = 1/32
    
    rx_symbols_corrected = np.zeros_like(rx_symbols)
    
    for l in range(NUM_SYMBOLS):
        # 1. 殘餘 CFO 校正 (k 從 -26..26)
        array_data = np.exp(1j * 2 * np.pi * l * ((FFT_SIZE + CP_SIZE_DATA) / FFT_SIZE) * epsilon * k_data)
        rx_symbols[:, l] = rx_symbols[:, l] * array_data
        
        array_pilot = np.exp(1j * 2 * np.pi * l * ((FFT_SIZE + CP_SIZE_DATA) / FFT_SIZE) * epsilon * k_pilot)
        Pilot[:, l] = Pilot[:, l] * array_pilot
        
        # 2. 共同相位錯誤 (CPE) 校正
        PQ = np.dot(np.conj(PILOT_TONES), Pilot[:, l]) # 修正順序
        beta = np.angle(PQ)
        rx_symbols[:, l] = rx_symbols[:, l] * np.exp(-1j * beta)
        
        # 3. 更新殘餘 CFO
        if l == 0:
            W[l] = np.dot(np.conj(PILOT_TONES), Pilot[:, l])
        else:
            W[l] = np.dot(np.conj(Pilot[:, l-1]), Pilot[:, l])
        
        if l > 2 and (l + 1) % 4 == 0: # 每 4 個符號更新一次
            V = W[l] + W[l-1] + W[l-2] + W[l-3]
            U = pho * V + (1 - pho) * U
            epsilon_r = np.angle(U) / (2 * np.pi * T_u * fc)
            epsilon = epsilon_0 + epsilon_r
        elif l < 3:
            epsilon_r = np.angle(W[l]) / (2 * np.pi * T_u * fc * (1 if l==0 else 1)) 
            epsilon = epsilon_0 + epsilon_r
            
        rx_symbols_corrected[:, l] = rx_symbols[:, l]
            
    return rx_symbols_corrected


# --- 3. 設定 USRP 接收器 (已更新 Serial) ---
print("Configuring USRP Receiver...")
SERIAL_RX = "34B733A"
rx_usrp = uhd.usrp.MultiUSRP(f"serial={SERIAL_RX}")

rx_usrp.set_rx_rate(FS)
rx_usrp.set_rx_freq(uhd.types.TuneRequest(FC))
rx_usrp.set_rx_gain(60) # MATLAB 中的 Gain (您可能需要調高此增益，例如 20 或 30)

# 設定 RX Streamer
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
rx_streamer = rx_usrp.get_rx_stream(stream_args)

# 接收的樣本數 (一次接收 1 秒的資料量)
samps_to_recv = int(FS) 
rcvdSignal = np.zeros(samps_to_recv, dtype=np.complex64)
rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))

# --- 4. 接收與處理迴圈 ---
print(f"Start receiving on serial {SERIAL_RX}...")
flag = False

while not flag:
    try:
        # 接收樣本
        metadata = uhd.types.RXMetadata()
        num_rx_samps = rx_streamer.recv(rcvdSignal, metadata)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(metadata.strerror())
            continue
            
        # 1. 對齊 (同步)
        sts_with_cp, _ = generate_preambles()
        frame_start_index, sync_result = sync(rcvdSignal, sts_with_cp[START_ZERO_LEN:])
        
        if sync_result:
            flag = True # 成功，跳出迴圈
            print("Sync Success!")
            
            # --- 繪圖：同步後的時域訊號 ---
            display_signal = rcvdSignal[frame_start_index : frame_start_index + FRAME_LEN]
            time_frame = np.arange(len(display_signal)) / FS * 1e6 # us
            
            plt.figure(figsize=(15, 7))
            plt.plot(time_frame, np.real(display_signal), '-k', label='Real Part')
            plt.title('Received Signal (Synchronized)')
            plt.xlabel('Time (µs)')
            plt.ylabel('Amplitude')
            
            sts_start_time = START_ZERO_LEN / FS * 1e6
            lts_start_time = (START_ZERO_LEN + STS_LEN) / FS * 1e6
            data_start_time = (START_ZERO_LEN + STS_LEN + LTS_LEN) / FS * 1e6
            data_end_time = (START_ZERO_LEN + STS_LEN + LTS_LEN + DATA_PAYLOAD_LEN) / FS * 1e6
            
            plt.axvline(sts_start_time, color='r', linestyle='--', label='STS Start')
            plt.axvline(lts_start_time, color='b', linestyle='--', label='LTS Start')
            plt.axvline(data_start_time, color='g', linestyle='--', label='Data Start')
            plt.axvline(data_end_time, color='m', linestyle='--', label='Data End')
            plt.legend()
            plt.grid(True)
            
            # 2. CFO 估測
            delta_f_total, rx_first_LTS = cfo_estimation(rcvdSignal, frame_start_index, FS)
            
            # 3. 通道估測 & 等化
            rx_LTS_symbols = np.fft.ifftshift(np.fft.fft(rx_first_LTS))
            equalization_const_64 = rx_LTS_symbols * L # H = Y/X, X=L, 1/L = L
            
            equalization_const = equalization_const_64[DATA_INDICES]
            equalization_const_for_pilot = equalization_const_64[PILOT_INDICES]
            
            # --- 繪圖：通道響應 ---
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.stem(np.abs(equalization_const_64))
            plt.title('Channel Magnitude')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.stem(np.angle(equalization_const_64))
            plt.title('Channel Phase')
            plt.ylabel('Phase (radians)')
            plt.xlabel('Subcarrier Index')
            plt.grid(True)
            
            # 4. 處理資料
            # (1) 提取資料部分
            data_start_idx = frame_start_index + START_ZERO_LEN + STS_LEN + LTS_LEN
            rx_data = rcvdSignal[data_start_idx : data_start_idx + DATA_PAYLOAD_LEN]
            
            # (2) 對資料做 CFO 校正 (時域)
            idx_lts1_start = frame_start_index + START_ZERO_LEN + STS_LEN + (LTS_LEN - 2 * FFT_SIZE) // 2
            t_data = np.arange(len(rx_data)) + (idx_lts1_start + FFT_SIZE) # 從 LTS 結束時開始
            rx_data_CFO = rx_data * np.exp((-1j * 2 * np.pi * delta_f_total * t_data) / FS)
            
            # (3) 取出符號 (Reshape, 移除 CP)
            rx_ofdm_with_cp = rx_data_CFO.reshape((NUM_SYMBOLS, SYMBOL_LEN_DATA)).T
            rx_ofdm = rx_ofdm_with_cp[CP_SIZE_DATA:, :] # 移除 CP (64x100)
            
            # (4) FFT, 提取 Data 和 Pilot
            rx_symbols, Pilot = ofdm_output(rx_ofdm)
            
            # --- 分析 1: 僅 LTS CFO 校正 (無等化) ---
            fig_const, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            rx_symbols_flat = rx_symbols.flatten()
            ax1.scatter(np.real(rx_symbols_flat), np.imag(rx_symbols_flat), s=5, alpha=0.5)
            ax1.set_title('Constellation (LTS CFO correction only)')
            ax1.set_xlabel('I'); ax1.set_ylabel('Q'); ax1.grid(True)
            
            rx_bits_1 = demap_16qam(rx_symbols_flat)
            num_errors_1 = np.sum(tx_bits != rx_bits_1)
            ber_1 = num_errors_1 / len(tx_bits)
            print(f'BER (LTS CFO correction only) : {ber_1:.5f} ({num_errors_1} errors)')

            # --- 分析 2: LTS CFO + 通道等化 ---
            rx_equalization = rx_symbols / equalization_const[:, np.newaxis]
            rx_eq_flat = rx_equalization.flatten()
            
            ax2.scatter(np.real(rx_eq_flat), np.imag(rx_eq_flat), s=5, alpha=0.5)
            ax2.set_title('Constellation (LTS correction + equalization)')
            ax2.set_xlabel('I'); ax2.set_ylabel('Q'); ax2.grid(True)
            
            rx_bits_2 = demap_16qam(rx_eq_flat)
            num_errors_2 = np.sum(tx_bits != rx_bits_2)
            ber_2 = num_errors_2 / len(tx_bits)
            print(f'BER (LTS correction + equalization): {ber_2:.5f} ({num_errors_2} errors)')
            
            # --- 分析 3: Pilot Tone 追蹤 (Bonus) ---
            Pilot_equalized = Pilot / equalization_const_for_pilot[:, np.newaxis]
            
            rx_symbols_piloted = pilot_tone_correction(
                rx_symbols.copy(), # 傳入副本
                Pilot_equalized.copy(),
                delta_f_total, 
                FC
            )
            
            rx_equalization_pilot = rx_symbols_piloted / equalization_const[:, np.newaxis]
            rx_pilot_flat = rx_equalization_pilot.flatten()
            
            ax3.scatter(np.real(rx_pilot_flat), np.imag(rx_pilot_flat), s=5, alpha=0.5)
            ax3.set_title('Constellation (With pilot tone correction)')
            ax3.set_xlabel('I'); ax3.set_ylabel('Q'); ax3.grid(True)
            
            rx_bits_3 = demap_16qam(rx_pilot_flat)
            num_errors_3 = np.sum(tx_bits != rx_bits_3)
            ber_3 = num_errors_3 / len(tx_bits)
            print(f'BER (With pilot tone correction): {ber_3:.5f} ({num_errors_3} errors)')
            
            plt.tight_layout()
            plt.show()

    except KeyboardInterrupt:
        print("\nStopping receiver...")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        break

# 停止串流
rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
print("Receiver stopped.")
import numpy as np
import uhd
import time

# --- 設定 ---
SERIAL_TX = "3475843"   # 您的 B210 序列號
TX_GAIN = 40.0          # 發射增益 (dB) - 可以適度調高
Fc = 5.1e9              # 載波頻率 (必須與 Rx 相同)
Fs = 1e6                # 取樣率 (1 MHz) - 降低 Fs 
                        #  - 讓我們有更多時間處理
# --- OFDM 參數 ---
FFT_size = 64
CP_len = 16
symbol_len = FFT_size + CP_len # 80

def create_preamble_sc(fft_size, cp_len):
    """ 
    創建 Schmidl-Cox (S&C) 同步前導碼 
    時域訊號的前半部(D)與後半部(D)相同 [D, D]
    """
    D = fft_size // 2 # 32
    
    # 只在偶數子載波上放置訊號
    preamble_freq = np.zeros(fft_size, dtype=np.complex64)
    preamble_freq[::2] = 1.0 + 0.0j # 簡單的 BPSK
    
    preamble_time = np.fft.ifft(preamble_freq)
    
    # 驗證 S&C 屬性 (前半部 == 後半部)
    # print(np.linalg.norm(preamble_time[:D] - preamble_time[D:])) # 應接近 0
    
    # 加上 CP
    preamble_with_cp = np.hstack([preamble_time[-cp_len:], preamble_time])
    return preamble_with_cp

def create_preamble_ltf(fft_size, cp_len):
    """ 
    創建 LTF (Long Training Field) 用於通道估測
    這只是一個隨機的 BPSK 序列 
    """
    # 確保 Tx 和 Rx 上的 "known" 序列相同
    rng = np.random.RandomState(seed=123) 
    
    # 在所有子載波上放置 BPSK (-1, 1)
    ltf_freq_known = rng.choice([-1, 1], fft_size)
    ltf_time = np.fft.ifft(ltf_freq_known)
    ltf_with_cp = np.hstack([ltf_time[-cp_len:], ltf_time])
    
    # 返回時域波形和頻域序列 (Rx 需要)
    return ltf_with_cp, ltf_freq_known

def create_data_symbols(num_symbols, fft_size, cp_len, num_data_carriers):
    """ 創建 QPSK 數據 """
    # 簡單的 QPSK
    qpsk_constellation = [1+1j, 1-1j, -1+1j, -1-1j] / np.sqrt(2)
    
    # 總共 N 個 symbol, 每個 symbol 有 M 個 data carrier
    data_bits_freq = np.random.choice(qpsk_constellation, (num_symbols, num_data_carriers))
    
    data_freq = np.zeros((num_symbols, fft_size), dtype=np.complex64)
    
    # 將數據放在中間的子載波上 (避開 DC 和邊緣)
    start_idx = (fft_size - num_data_carriers) // 2
    end_idx = start_idx + num_data_carriers
    data_freq[:, start_idx:end_idx] = data_bits_freq
    
    data_time = np.fft.ifft(data_freq, axis=1)
    
    # 加上 CP
    cp = data_time[:, -cp_len:]
    data_with_cp = np.hstack([cp, data_time])
    
    return data_with_cp.flatten(), data_bits_freq[:, :num_data_carriers]

# --- 1. 產生訊框 (Frame) ---
print("Generating simple OFDM frame...")

# 符號 0: S&C Preamble (用於同步 + CFO 估計)
preamble_sc = create_preamble_sc(FFT_size, CP_len) # 80 samples

# 符號 1: LTF Preamble (用於通道估測)
preamble_ltf, ltf_freq_known = create_preamble_ltf(FFT_size, CP_len) # 80 samples
# (儲存 LTF 頻域序列，以便 Rx 讀取)
np.savez('ltf_data.npz', ltf_freq_known=ltf_freq_known)

# 符號 2-11: 10 個數據符號
num_data_carriers = 48 # 64 個子載波中，用 48 個來傳數據
data_payload, gt_payload_freq = create_data_symbols(10, FFT_size, CP_len, num_data_carriers) # 800 samples
# (儲存 Ground Truth 數據，以便 Rx 讀取)
np.savez('gt_simple.npz', gt_payload_freq=gt_payload_freq)


# 組合成一個完整的訊框
tx_waveform = np.hstack([preamble_sc, preamble_ltf, data_payload])
N_SAMPLES_PER_FRAME = len(tx_waveform) # 80 + 80 + 800 = 960 samples
print(f"Frame generated. Total samples: {N_SAMPLES_PER_FRAME}")

# 功率正規化
tx_waveform = tx_waveform / np.max(np.abs(tx_waveform)) * 0.5

# --- 2. 連接 USRP ---
print(f"Connecting to TX USRP (B210) at serial={SERIAL_TX}...")
usrp_tx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_TX}"))

# --- 3. 設定 USRP ---
usrp_tx.set_clock_source("internal", 0)
usrp_tx.set_time_source("internal", 0)
usrp_tx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
usrp_tx.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:A"), 0)
usrp_tx.set_tx_antenna("TX/RX", 0)
usrp_tx.set_tx_rate(Fs)
usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_tx.set_tx_gain(TX_GAIN, 0)
print(f"B210 (Tx) setup complete. Rate: {Fs/1e6} MHz, Freq: {Fc/1e9} GHz, Gain: {TX_GAIN} dB")

# --- 4. 建立串流 ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
tx_streamer = usrp_tx.get_tx_stream(stream_args)

# --- 5. 連續發射 (只發射 5 秒鐘) ---
print("\n*** Starting finite transmission (5 seconds)... ***")
tx_metadata = uhd.types.TXMetadata()
tx_metadata.start_of_burst = True
tx_metadata.end_of_burst = False
tx_metadata.has_time_spec = True

frame_duration = N_SAMPLES_PER_FRAME / Fs
total_duration = 5.0 # 秒
num_frames_to_send = int(total_duration / frame_duration)

next_tx_time = usrp_tx.get_time_now().get_real_secs() + 0.2

try:
    for i in range(num_frames_to_send):
        tx_metadata.time_spec = uhd.types.TimeSpec(next_tx_time)
        tx_streamer.send(tx_waveform, tx_metadata)
        next_tx_time += frame_duration
        
        if i % 100 == 0:
            print(f"Sent frame {i}/{num_frames_to_send}")

except KeyboardInterrupt:
    print("\nStopping transmission early...")

finally:
    tx_metadata.end_of_burst = True
    tx_streamer.send(np.zeros_like(tx_waveform), tx_metadata)
    print("Transmitter shut down.")
import numpy as np
import uhd
import time

# --- 設定 ---
SERIAL_TX = "3475843"   # 你的 B210/Tx 序列號 (請確認)
TX_GAIN = 60.0          # 發射增益 (dB) - 我們可以調高一點
Fc = 5.4e9              # 載波頻率 (必須與 Rx 相同)
Fs = 1e6                # 取樣率 (1 MHz)

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
    preamble_freq = np.zeros(fft_size, dtype=np.complex64)
    preamble_freq[::2] = 1.0 + 0.0j # 簡單的 BPSK
    preamble_time = np.fft.ifft(preamble_freq)
    preamble_with_cp = np.hstack([preamble_time[-cp_len:], preamble_time])
    return preamble_with_cp

def create_preamble_ltf(fft_size, cp_len):
    """ 
    創建 LTF (Long Training Field) 用於通道估測
    """
    rng = np.random.RandomState(seed=123) # 確保 Tx/Rx 上的序列相同
    ltf_freq_known = rng.choice([-1, 1], fft_size)
    ltf_time = np.fft.ifft(ltf_freq_known)
    ltf_with_cp = np.hstack([ltf_time[-cp_len:], ltf_time])
    return ltf_with_cp, ltf_freq_known

def create_data_symbols(num_symbols, fft_size, cp_len, num_data_carriers):
    """ 創建 QPSK 數據 """
    qpsk_constellation = [1+1j] / np.sqrt(2)
    data_bits_freq = np.random.choice(qpsk_constellation, (num_symbols, num_data_carriers))
    
    data_freq = np.zeros((num_symbols, fft_size), dtype=np.complex64)
    start_idx = (fft_size - num_data_carriers) // 2
    end_idx = start_idx + num_data_carriers
    data_freq[:, start_idx:end_idx] = data_bits_freq
    
    data_time = np.fft.ifft(data_freq, axis=1)
    cp = data_time[:, -cp_len:]
    data_with_cp = np.hstack([cp, data_time])
    
    # 返回扁平化的時域波形, 和「未扁平化」的頻域 GT
    return data_with_cp.flatten(), data_freq[:, start_idx:end_idx]

# --- 1. 產生訊框 (Frame) ---
print("Generating simple OFDM frame...")

# 符號 0: S&C Preamble (用於同步 + CFO 估計)
preamble_sc = create_preamble_sc(FFT_size, CP_len) # 80 samples

# 符號 1: LTF Preamble (用於通道估測)
preamble_ltf, ltf_freq_known = create_preamble_ltf(FFT_size, CP_len) # 80 samples
np.savez('ltf_data.npz', ltf_freq_known=ltf_freq_known)

# 符號 2-11: 10 個數據符號
num_data_carriers = 48 # 64 個子載波中，用 48 個來傳數據
data_payload, gt_payload_freq = create_data_symbols(10, FFT_size, CP_len, num_data_carriers) # 800 samples
np.savez('gt_simple.npz', gt_payload_freq=gt_payload_freq)


# 組合成一個完整的訊框
tx_waveform = np.hstack([preamble_sc, preamble_ltf, data_payload])
N_SAMPLES_PER_FRAME = len(tx_waveform) # 80 + 80 + 800 = 960 samples
print(f"Frame generated. Total samples: {N_SAMPLES_PER_FRAME}")

# 功率正規化
tx_waveform = tx_waveform / np.max(np.abs(tx_waveform)) * 0.5
tx_waveform = tx_waveform.astype(np.complex64).reshape(1, -1) # (1, 960)

# --- 2. 連接 USRP ---
print(f"Connecting to TX USRP (B210) at serial={SERIAL_TX}...")
usrp_tx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_TX}"))

# (關鍵) 使用內部時脈
usrp_tx.set_clock_source("internal")
usrp_tx.set_time_source("internal")
usrp_tx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
usrp_tx.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0"), 0)
usrp_tx.set_tx_antenna("TX/RX", 0)
usrp_tx.set_tx_rate(Fs)
usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_tx.set_tx_gain(TX_GAIN, 0)
print(f"B210 (Tx) setup complete. Rate: {Fs/1e6} MHz, Freq: {Fc/1e9} GHz, Gain: {TX_GAIN} dB")

# --- 4. 建立串流 ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
tx_streamer = usrp_tx.get_tx_stream(stream_args)
tx_metadata = uhd.types.TXMetadata()
tx_metadata.start_of_burst = False
tx_metadata.end_of_burst = False
tx_metadata.has_time_spec = False # 連續模式

# --- 5. 連續發射 ---
print("\n*** Starting continuous transmission... (Press Ctrl+C to stop) ***")
try:
    while True:
        tx_streamer.send(tx_waveform, tx_metadata)
except KeyboardInterrupt:
    print("\nStopping transmission early...")
finally:
    tx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    print("Transmitter shut down.")
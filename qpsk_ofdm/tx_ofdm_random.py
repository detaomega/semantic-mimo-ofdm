import numpy as np
import uhd
import time

# --- 設定 ---
SERIAL_TX = "3475843"   # 你的 B210/Tx 序列號 (請確認)
TX_GAIN = 40.0
Fc = 2.0e9
Fs = 1e6
GT_FILE_LTF = 'ltf_data.npz'
GT_FILE_PAYLOAD = 'gt_simple.npz'

# --- OFDM 參數 ---
FFT_size = 64
CP_len = 16
symbol_len = FFT_size + CP_len # 80
num_data_carriers = 48
num_data_symbols = 10

# --- 1. Preamble 函式 (保持不變) ---
def create_preamble_sc(fft_size, cp_len):
    D = fft_size // 2
    preamble_freq = np.zeros(fft_size, dtype=np.complex64)
    preamble_freq[::2] = 1.0 + 0.0j
    preamble_time = np.fft.ifft(preamble_freq)
    preamble_with_cp = np.hstack([preamble_time[-cp_len:], preamble_time])
    return preamble_with_cp

def create_preamble_ltf(fft_size, cp_len):
    rng = np.random.RandomState(seed=123) # 保持種子固定
    ltf_freq_known = rng.choice([-1, 1], fft_size)
    ltf_time = np.fft.ifft(ltf_freq_known)
    ltf_with_cp = np.hstack([ltf_time[-cp_len:], ltf_time])
    return ltf_with_cp, ltf_freq_known

# --- 2. (!!! 關鍵修改 !!!) 產生「任意」 Complex Number ---
def create_arbitrary_data(num_symbols, fft_size, cp_len, num_data_carriers):
    """ 
    創建「任意」 complex number (例如 0.4 + 0.3j)
    而不是 QPSK
    """
    print("Generating ARBITRARY complex payload...")
    
    # 1. 隨機產生 I 和 Q 值 (範圍在 -1.0 到 +1.0 之間)
    I_values = (np.random.rand(num_symbols, num_data_carriers) - 0.5) * 2.0
    Q_values = (np.random.rand(num_symbols, num_data_carriers) - 0.5) * 2.0
    
    # 2. 這就是我們的「任意 complex number」
    arbitrary_payload_freq = (I_values + 1j*Q_values).astype(np.complex64)
    
    # 範例：檢查一個值 (可能就像 0.4 + 0.3j)
    print(f"Example arbitrary complex number: {arbitrary_payload_freq[0,0]}")

    # --- (以下與之前相同) ---
    
    # 3. 放置到子載波上
    data_freq = np.zeros((num_symbols, fft_size), dtype=np.complex64)
    start_idx = (fft_size - num_data_carriers) // 2
    end_idx = start_idx + num_data_carriers
    data_freq[:, start_idx:end_idx] = arbitrary_payload_freq
    
    # 4. IFFT
    data_time = np.fft.ifft(data_freq, axis=1)
    
    # 5. 加 CP
    cp = data_time[:, -cp_len:]
    data_with_cp = np.hstack([cp, data_time])
    
    # 返回時域波形, 和「頻域」的 Ground Truth
    return data_with_cp.flatten(), arbitrary_payload_freq

# --- 3. 產生訊框 (Frame) ---
print("Generating simple OFDM frame...")

preamble_sc = create_preamble_sc(FFT_size, CP_len)
preamble_ltf, ltf_freq_known = create_preamble_ltf(FFT_size, CP_len)
# (儲存 LTF，Rx 需要它)
np.savez(GT_FILE_LTF, ltf_freq_known=ltf_freq_known)

# (!!! 關鍵修改 !!!)
# 呼叫我們的新函式
data_payload, gt_payload_freq = create_arbitrary_data(
    num_data_symbols, FFT_size, CP_len, num_data_carriers
)
# (儲存「任意」的 Ground Truth，Rx 需要它)
np.savez(GT_FILE_PAYLOAD, gt_payload_freq=gt_payload_freq)
print(f"Arbitrary Ground Truth payload saved to {GT_FILE_PAYLOAD}")


# 組合成一個完整的訊框
tx_waveform = np.hstack([preamble_sc, preamble_ltf, data_payload])
N_SAMPLES_PER_FRAME = len(tx_waveform) # 960 samples

# 功率正規化
tx_waveform = tx_waveform / np.max(np.abs(tx_waveform)) * 0.5
tx_waveform = tx_waveform.astype(np.complex64).reshape(1, -1) # (1, 960)

# --- 4. 連接 USRP ---
print(f"Connecting to TX USRP (B210) at serial={SERIAL_TX}...")
usrp_tx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_TX}"))
usrp_tx.set_clock_source("internal")
usrp_tx.set_time_source("internal")
usrp_tx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
usrp_tx.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:A"), 0)
usrp_tx.set_tx_antenna("TX/RX", 0)
usrp_tx.set_tx_rate(Fs)
usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_tx.set_tx_gain(TX_GAIN, 0)
print(f"B210 (Tx) setup complete. Rate: {Fs/1e6} MHz, Freq: {Fc/1e9} GHz, Gain: {TX_GAIN} dB")

# --- 5. 建立串流 ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
tx_streamer = usrp_tx.get_tx_stream(stream_args)
tx_metadata = uhd.types.TXMetadata()
tx_metadata.start_of_burst = False
tx_metadata.end_of_burst = False
tx_metadata.has_time_spec = False 

# --- 6. 連續發射 (重複發射同一個「任意」訊框) ---
print("\n*** Starting continuous transmission... (Press Ctrl+C to stop) ***")
try:
    while True:
        tx_streamer.send(tx_waveform, tx_metadata)
except KeyboardInterrupt:
    print("\nStopping transmission early...")
finally:
    tx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    print("Transmitter shut down.")
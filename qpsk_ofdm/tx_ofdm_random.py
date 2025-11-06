import numpy as np
import uhd
import time

# --- 設定 ---
SERIAL_TX = "3475843"   # 你的 B210/Tx 序列號 (請確認)
TX_GAIN = 40.0
Fc = 5.4e9
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

def create_ltf_time(fft_size, cp_len, ltf_freq_known):
    """ (修改) 從已知的頻域序列產生時域波形 """
    ltf_time = np.fft.ifft(ltf_freq_known)
    ltf_with_cp = np.hstack([ltf_time[-cp_len:], ltf_time])
    return ltf_with_cp

def create_data_time(fft_size, cp_len, num_data_carriers, arbitrary_payload_freq):
    """ (修改) 從已知的頻域序列產生時域波形 """
    num_symbols = arbitrary_payload_freq.shape[0]
    
    data_freq = np.zeros((num_symbols, fft_size), dtype=np.complex64)
    start_idx = (fft_size - num_data_carriers) // 2
    end_idx = start_idx + num_data_carriers
    data_freq[:, start_idx:end_idx] = arbitrary_payload_freq
    
    data_time = np.fft.ifft(data_freq, axis=1)
    cp = data_time[:, -cp_len:]
    data_with_cp = np.hstack([cp, data_time])
    
    return data_with_cp.flatten()

# --- 2. (!!! 關鍵修改 !!!) 從檔案讀取 Ground Truth ---
print("Loading Ground Truth files...")
try:
    ltf_data = np.load(GT_FILE_LTF)
    ltf_freq_known = ltf_data['ltf_freq_known']
    
    gt_data = np.load(GT_FILE_PAYLOAD)
    gt_payload_freq = gt_data['gt_payload_freq']
except FileNotFoundError:
    print(f"!!! 錯誤: 找不到 {GT_FILE_LTF} 或 {GT_FILE_PAYLOAD} !!!")
    print("請先執行 1_generate_truth_files.py")
    exit()
print("Ground Truth files loaded.")

# --- 3. 產生訊框 (Frame) (僅執行一次) ---
print("Generating simple OFDM frame...")

preamble_sc = create_preamble_sc(FFT_size, CP_len)
preamble_ltf = create_ltf_time(FFT_size, CP_len, ltf_freq_known)
data_payload = create_data_time(
    FFT_size, CP_len, num_data_carriers, gt_payload_freq
)

# 組合成一個完整的訊框
tx_waveform = np.hstack([preamble_sc, preamble_ltf, data_payload])
N_SAMPLES_PER_FRAME = len(tx_waveform) # 960 samples

# 功率正規化
tx_waveform = tx_waveform / np.max(np.abs(tx_waveform)) * 0.5
tx_waveform = tx_waveform.astype(np.complex64).reshape(1, -1) # (1, 960)
print(f"Waveform ready. Total samples: {N_SAMPLES_PER_FRAME}")

# --- 4. 連接 USRP ---
print(f"Connecting to TX USRP (B210) at serial={SERIAL_TX}...")
usrp_tx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_TX}"))
usrp_tx.set_clock_source("internal")
usrp_tx.set_time_source("internal")
usrp_tx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
usrp_tx.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0"), 0)
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

# --- 6. 連續發射 (重複發射同一個「固定」訊框) ---
print("\n*** Starting continuous transmission... (Press Ctrl+C to stop) ***")
try:
    while True:
        tx_streamer.send(tx_waveform, tx_metadata)
except KeyboardInterrupt:
    print("\nStopping transmission early...")
finally:
    tx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    print("Transmitter shut down.")
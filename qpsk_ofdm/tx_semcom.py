import numpy as np
import uhd
import time
import sys

# --- 設定 ---
SERIAL_TX = "3475843"   # 你的 B210/Tx 序列號
TX_GAIN = 70.0          # 根據實驗距離調整，太近可以調低 (e.g. 50)
Fc = 5.4e9
Fs = 1e6

# 檔案路徑
GT_FILE_LTF = 'ltf_data.npz'
GT_FILE_PAYLOAD = 'transmit_payload.npy' # 這是 JSCC sender 產出的 .npy

# --- OFDM 參數 ---
FFT_size = 64
CP_len = 16
num_data_carriers = 48 # 有效子載波數

# (!!! 關鍵修改: DC Nulls !!!)
# 在 64 個 FFT 點的中央空出 8 個子載波來避開 DC 濾波器
DC_NULLS = 8 
num_data_carriers_per_side = num_data_carriers // 2 # 24

# 計算子載波的索引
total_used_block = num_data_carriers + DC_NULLS # 56
data_block_start = (FFT_size - total_used_block) // 2 # 4

data_left_start = data_block_start              # 4
data_left_end = data_left_start + num_data_carriers_per_side # 28

data_right_start = data_left_end + DC_NULLS     # 36
data_right_end = data_right_start + num_data_carriers_per_side # 60

# --- 1. Preamble 函式 ---
def create_preamble_sc(fft_size, cp_len):
    preamble_freq = np.zeros(fft_size, dtype=np.complex64)
    preamble_freq[::2] = 1.0 + 0.0j # 每隔一個放一個導頻
    preamble_time = np.fft.ifft(preamble_freq)
    preamble_with_cp = np.hstack([preamble_time[-cp_len:], preamble_time])
    return preamble_with_cp

def create_ltf_time(fft_size, cp_len, ltf_freq_known):
    ltf_time = np.fft.ifft(ltf_freq_known)
    ltf_with_cp = np.hstack([ltf_time[-cp_len:], ltf_time])
    return ltf_with_cp

def create_data_time(fft_size, cp_len, payload_freq_matrix):
    """ 
    將 payload (Shape: [N_sym, 48]) 映射到 OFDM 子載波 
    """
    num_symbols = payload_freq_matrix.shape[0]
    data_freq = np.zeros((num_symbols, fft_size), dtype=np.complex64)
    
    # 映射左半邊 (前24個) -> Index 4~27
    data_freq[:, data_left_start:data_left_end] = payload_freq_matrix[:, :num_data_carriers_per_side]
    
    # 映射右半邊 (後24個) -> Index 36~59
    data_freq[:, data_right_start:data_right_end] = payload_freq_matrix[:, num_data_carriers_per_side:]
    
    # IFFT (沿著最後一個維度做)
    data_time = np.fft.ifft(data_freq, axis=1)
    
    # 加 CP
    cp = data_time[:, -cp_len:]
    data_with_cp = np.hstack([cp, data_time])
    
    # 展平變成一維時間序列
    return data_with_cp.flatten()

# --- 2. 從檔案讀取 Ground Truth ---
print("Loading files...")
try:
    # 讀取 LTF
    ltf_data = np.load(GT_FILE_LTF)
    # 注意：這裡要確認你的 ltf_data.npz 裡的 key 是什麼，通常是 'ltf_freq_known' 或 'arr_0'
    if 'ltf_freq_known' in ltf_data:
        ltf_freq_known = ltf_data['ltf_freq_known']
    else:
        ltf_freq_known = ltf_data['arr_0'] # Fallback

    # 讀取 Payload
    raw_data = np.load(GT_FILE_PAYLOAD) # shape (512, 2)
    
    # 轉為複數
    flat_complex_data = raw_data[:, 0] + 1j * raw_data[:, 1] # shape (512,)
    
    print(f"Loaded payload. Total symbols: {len(flat_complex_data)}")

    # --- (重要) 資料整形與 Padding ---
    # 我們有 512 個數據，每個 OFDM Symbol 能載 48 個
    # 512 / 48 = 10.66 -> 需要 11 個 Symbol
    n_data = len(flat_complex_data)
    n_symbols = int(np.ceil(n_data / num_data_carriers)) # 11
    n_padding = n_symbols * num_data_carriers - n_data   # 11*48 - 512 = 16
    
    print(f"Reshaping: Requires {n_symbols} OFDM symbols (Padding {n_padding} zeros)")
    
    # 補零
    padded_data = np.pad(flat_complex_data, (0, n_padding), 'constant')
    
    # Reshape 成 (11, 48)
    payload_freq_matrix = padded_data.reshape(n_symbols, num_data_carriers)

except FileNotFoundError:
    print(f"!!! 錯誤: 找不到檔案 !!! 請確認路徑")
    sys.exit(1)
except Exception as e:
    print(f"!!! 資料處理錯誤: {e}")
    sys.exit(1)

# --- 3. 產生訊框 (Frame) ---
print("Generating OFDM frame...")

# 生成各部分
preamble_sc = create_preamble_sc(FFT_size, CP_len)
preamble_ltf = create_ltf_time(FFT_size, CP_len, ltf_freq_known)
data_waveform = create_data_time(FFT_size, CP_len, payload_freq_matrix)

# 組合 Frame: [STS] + [LTS] + [DATA]
# 這裡只用了一個 Preamble (S&C) + LTF
tx_waveform = np.hstack([preamble_sc, preamble_ltf, data_waveform])

# 正規化振幅 (避免 USRP Clipping)
max_amp = np.max(np.abs(tx_waveform))
if max_amp > 0:
    tx_waveform = tx_waveform / max_amp * 0.7 # 稍微留點 Headroom
else:
    print("警告: 波形全為 0")

# 轉換型態給 UHD (complex64)
tx_waveform = tx_waveform.astype(np.complex64).reshape(1, -1)
print(f"Waveform ready. Total samples: {tx_waveform.shape[1]}")

# --- 4. 連接 USRP ---
print(f"Connecting to TX USRP (B210) at serial={SERIAL_TX}...")
try:
    usrp_tx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_TX}"))
    usrp_tx.set_clock_source("internal")
    usrp_tx.set_tx_rate(Fs)
    usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
    usrp_tx.set_tx_gain(TX_GAIN, 0)
    
    # 設置天線 (通常是 "TX/RX")
    usrp_tx.set_tx_antenna("TX/RX", 0)
    
    print(f"Tx Setup: Freq={Fc/1e9}GHz, Gain={TX_GAIN}dB, Rate={Fs/1e6}MHz")
except RuntimeError as e:
    print(f"無法連接 USRP: {e}")
    sys.exit(1)

# --- 5. 建立串流 ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
tx_streamer = usrp_tx.get_tx_stream(stream_args)
tx_metadata = uhd.types.TXMetadata()
tx_metadata.start_of_burst = False
tx_metadata.end_of_burst = False

# --- 6. 連續發射 ---
print("\n*** Starting continuous transmission... (Press Ctrl+C to stop) ***")
try:
    while True:
        # send 回傳發送的樣本數
        num_sent = tx_streamer.send(tx_waveform, tx_metadata)
except KeyboardInterrupt:
    print("\nStopping transmission...")
finally:
    # 停止指令
    tx_metadata.end_of_burst = True
    tx_streamer.send(np.zeros((1, 100), dtype=np.complex64), tx_metadata)
    print("Done.")
import numpy as np
import uhd
import time
from ofdm_parameters import * # 導入所有共用參數

# --- 1. 產生資料承載 (Data Payload) ---
def generate_data_payload():
    num_bits = 48 * LOG2_M * NUM_SYMBOLS
    tx_bits = np.random.randint(0, 2, num_bits, dtype=np.uint8)
    
    # 將 bits 調變為 16-QAM
    tx_symbols = map_16qam(tx_bits)
    
    # 重塑為 (48, num_symbols)
    tx_symbols_reshaped = tx_symbols.reshape((NUM_SYMBOLS, 48)).T
    
    # 建立 OFDM 符號 (頻域)
    tx_frame = np.zeros((FFT_SIZE, NUM_SYMBOLS), dtype=np.complex64)
    
    # 映射資料
    tx_frame[DATA_INDICES, :] = tx_symbols_reshaped
    # 插入導頻
    tx_frame[PILOT_INDICES, :] = np.tile(PILOT_TONES[:, np.newaxis], (1, NUM_SYMBOLS))
    
    # IFFT (沿著子載波維度)
    tx_ofdm = np.fft.ifft(np.fft.fftshift(tx_frame, axes=0), axis=0) * FFT_SIZE
    
    # 加上 CP (16 samples)
    tx_ofdm_with_cp = np.concatenate([tx_ofdm[FFT_SIZE - CP_SIZE_DATA:, :], tx_ofdm], axis=0)
    
    # 展平為 1D 向量
    return tx_ofdm_with_cp.T.flatten(), tx_bits

# --- 2. 組合完整封包 (Frame) ---
print("Generating preamble and data...")
sts_with_cp, lts_with_cp = generate_preambles()
data_payload, tx_bits = generate_data_payload() # 修正變數名稱

# 儲存 tx_bits 以便 RX 計算 BER
np.save('tx_bits.npy', tx_bits)
print("tx_bits saved to tx_bits.npy")

# 功率正規化 (如 MATLAB 腳本)
sts_power = np.sqrt(np.mean(np.abs(sts_with_cp)**2))
lts_power = np.sqrt(np.mean(np.abs(lts_with_cp)**2))
data_power = np.sqrt(np.mean(np.abs(data_payload)**2)) # 修正

lts_normalized = lts_with_cp / (lts_power / sts_power)
data_normalized = data_payload / (data_power / sts_power) # 修正

# 組合
start_zero = np.zeros(START_ZERO_LEN, dtype=np.complex64)
end_zero = np.zeros(END_ZERO_LEN, dtype=np.complex64)

frame = np.concatenate([start_zero, sts_with_cp, lts_normalized, data_normalized, end_zero])
# 重複 3 次 (如 MATLAB 腳本)
data_to_tx = np.tile(frame, 3).astype(np.complex64)

print(f"Frame length: {len(frame)}")
print(f"Total data to transmit: {len(data_to_tx)} samples")

# --- 3. 設定 USRP 發送器 (已更新 Serial) ---
print("Configuring USRP Transmitter...")
SERIAL_TX = "3475843"
tx_usrp = uhd.usrp.MultiUSRP(f"serial={SERIAL_TX}")

tx_usrp.set_tx_rate(FS)
tx_usrp.set_tx_freq(uhd.types.TuneRequest(FC))
tx_usrp.set_tx_gain(60) # MATLAB 中的 Gain

# 設定 TX Streamer
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
tx_streamer = tx_usrp.get_tx_stream(stream_args)

# 發送元數據
metadata = uhd.types.TXMetadata()
metadata.start_of_burst = False
metadata.has_time_spec = False # 連續發送

# --- 4. 發送迴圈 ---
print(f"Start transmitting on serial {SERIAL_TX}...")
try:
    while True:
        tx_streamer.send(data_to_tx, metadata)
        # 保持安靜，避免洗版
        # print("Frame sent")
        time.sleep(0.1) # 迴圈間隔
except KeyboardInterrupt:
    print("\nStopping transmitter...")
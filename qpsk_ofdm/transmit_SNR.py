# transmit.py
import numpy as np
import uhd
import time
import os

# --- 設定區 ---
SERIAL_TX = "3475843"   # 請確認你的 B210 序號
TX_GAIN = 70.0          # 發射增益
Fc = 5.2e9              # 中心頻率 5.2 GHz
Fs = 1e6                # 取樣率 1 MHz

# --- OFDM 參數 ---
FFT_size = 64
CP_len = 16
num_data_symbols = 10
num_data_carriers = 48  # 實際使用的資料子載波數
DC_NULLS = 8            # DC 附近的空子載波數

# --- 子載波映射 (關鍵：定義哪裡發射訊號，哪裡留白) ---
num_data_carriers_per_side = num_data_carriers // 2 # 24
total_used_block = num_data_carriers + DC_NULLS     # 56
data_block_start = (FFT_size - total_used_block) // 2 # 4

# 定義資料區塊索引
idx_data_left = np.arange(data_block_start, data_block_start + num_data_carriers_per_side)
idx_data_right = np.arange(data_block_start + num_data_carriers_per_side + DC_NULLS, 
                           data_block_start + total_used_block)

# --- 自動產生測試資料 (避免找不到檔案報錯) ---
def generate_test_data():
    print("Generating random test data...")
    # 1. Preamble (Schmidl-Cox)
    preamble_freq = np.zeros(FFT_size, dtype=np.complex64)
    preamble_freq[::2] = 1.0 + 0.0j # 頻域間隔
    preamble_time = np.fft.ifft(preamble_freq)
    preamble_cp = np.hstack([preamble_time[-CP_len:], preamble_time])
    
    # 2. LTF
    ltf_freq = np.random.choice([1, -1], FFT_size).astype(np.complex64)
    ltf_time = np.fft.ifft(ltf_freq)
    ltf_cp = np.hstack([ltf_time[-CP_len:], ltf_time])
    
    # 3. Payload Data
    # 產生 QPSK 符號
    payload_bits = np.random.randint(0, 2, (num_data_symbols, num_data_carriers * 2))
    payload_qpsk = (2*payload_bits[:,::2]-1) + 1j*(2*payload_bits[:,1::2]-1)
    payload_qpsk *= 1/np.sqrt(2) # 正規化能量
    
    return preamble_cp, ltf_cp, payload_qpsk, ltf_freq

# 嘗試讀取檔案，若無則生成
try:
    ltf_data = np.load('ltf_data.npz')
    gt_data = np.load('gt_simple.npz')
    ltf_freq_known = ltf_data['ltf_freq_known']
    gt_payload_freq = gt_data['gt_payload_freq']
    
    # 重建 Preamble/LTF
    preamble_sc = np.fft.ifft(np.zeros(FFT_size, dtype=np.complex64)) # 簡化 placeholder
    preamble_sc[::2] = 1
    preamble_sc = np.fft.ifft(preamble_sc)
    preamble_cp = np.hstack([preamble_sc[-CP_len:], preamble_sc])
    
    ltf_time = np.fft.ifft(ltf_freq_known)
    ltf_cp = np.hstack([ltf_time[-CP_len:], ltf_time])
    
    payload_src = gt_payload_freq
    print("Loaded existing GT files.")
except:
    print("GT files not found, using generated data.")
    preamble_cp, ltf_cp, payload_src, ltf_freq_known = generate_test_data()


# --- 構建發射波形 ---
def create_tx_waveform(payload_freq):
    tx_signal = []
    # 加入 Preamble
    tx_signal.append(preamble_cp)
    # 加入 LTF
    tx_signal.append(ltf_cp)
    
    # 加入 Data Symbols
    for i in range(num_data_symbols):
        symbol_freq = np.zeros(FFT_size, dtype=np.complex64)
        # 映射左半邊
        symbol_freq[idx_data_left] = payload_freq[i, :num_data_carriers_per_side]
        # 映射右半邊
        symbol_freq[idx_data_right] = payload_freq[i, num_data_carriers_per_side:]
        
        symbol_time = np.fft.ifft(symbol_freq)
        symbol_cp = np.hstack([symbol_time[-CP_len:], symbol_time])
        tx_signal.append(symbol_cp)
        
    return np.hstack(tx_signal)

# 產生最終波形
tx_waveform = create_tx_waveform(payload_src)
# 能量正規化，避免 DAC 過載
tx_waveform = tx_waveform / np.max(np.abs(tx_waveform)) * 0.5
tx_waveform = tx_waveform.astype(np.complex64).reshape(1, -1)

print(f"Tx Waveform prepared. Length: {tx_waveform.shape[1]} samples")

# --- USRP 發射主程式 ---
def main():
    print(f"Connecting to TX USRP ({SERIAL_TX})...")
    usrp_tx = uhd.usrp.MultiUSRP(f"serial={SERIAL_TX}")
    
    usrp_tx.set_tx_rate(Fs)
    usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
    usrp_tx.set_tx_gain(TX_GAIN, 0)
    
    # 設置串流
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp_tx.get_tx_stream(st_args)
    metadata = uhd.types.TXMetadata()
    
    print(f"Transmitting... Freq: {Fc/1e9}GHz, Gain: {TX_GAIN}dB")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # 持續發射同一幀訊號
            tx_streamer.send(tx_waveform, metadata)
    except KeyboardInterrupt:
        print("\nStopping transmission...")
        tx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

if __name__ == "__main__":
    main()
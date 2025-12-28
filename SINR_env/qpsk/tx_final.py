import numpy as np
import uhd
import time
import argparse

# --- 參數設定 ---
# 請修改您的 TX 序號
SERIAL_TX = "3475843"   
TX_GAIN = 60.0          
Fc = 5.4e9              
Fs = 1e6                

# --- OFDM 參數 ---
FFT_size = 64
CP_len = 16

# --- Pilot 設定 (仿照 WiFi 規格) ---
ALL_CARRIERS = np.arange(-26, 27)
ALL_CARRIERS = ALL_CARRIERS[ALL_CARRIERS != 0] # 去除直流

PILOT_CARRIERS = np.array([-21, -7, 7, 21]) # Pilot 位置
PILOT_VALUE = 1.0 + 0j 

DATA_CARRIERS = np.setdiff1d(ALL_CARRIERS, PILOT_CARRIERS)
N_DATA = len(DATA_CARRIERS)   # 48

def create_preamble_sc(fft_size, cp_len):
    """ Schmidl-Cox 同步前導碼 """
    preamble_freq = np.zeros(fft_size, dtype=np.complex64)
    preamble_freq[::2] = 1.0 + 0.0j 
    preamble_time = np.fft.ifft(preamble_freq)
    return np.hstack([preamble_time[-cp_len:], preamble_time])

def create_preamble_ltf(fft_size, cp_len):
    """ LTF 通道估測前導碼 """
    rng = np.random.RandomState(seed=123) 
    ltf_freq = np.zeros(fft_size, dtype=np.complex64)
    valid_idx = (ALL_CARRIERS + fft_size) % fft_size
    ltf_freq[valid_idx] = rng.choice([-1, 1], len(valid_idx))
    ltf_time = np.fft.ifft(ltf_freq)
    return np.hstack([ltf_time[-cp_len:], ltf_time]), ltf_freq

def create_data_with_pilots(num_symbols, fft_size, cp_len):
    """ 產生帶有 Pilot 的數據符號 """
    # QPSK 數據
    qpsk_const = [1+1j, -1+1j, -1-1j, 1-1j]
    data_bits = np.random.choice(qpsk_const, (num_symbols, N_DATA)) / np.sqrt(2)
    
    sym_freq = np.zeros((num_symbols, fft_size), dtype=np.complex64)
    
    # 映射
    data_idx = (DATA_CARRIERS + fft_size) % fft_size
    pilot_idx = (PILOT_CARRIERS + fft_size) % fft_size
    
    sym_freq[:, data_idx] = data_bits
    sym_freq[:, pilot_idx] = PILOT_VALUE
    
    # IFFT
    sym_time = np.fft.ifft(sym_freq, axis=1)
    cp = sym_time[:, -cp_len:]
    sym_with_cp = np.hstack([cp, sym_time])
    
    return sym_with_cp.flatten(), data_bits

def transmit(serial):
    print(f"--- OFDM 發射機 (Pilot版) ---")
    
    # 1. 產生訊號
    preamble_sc = create_preamble_sc(FFT_size, CP_len)
    preamble_ltf, ltf_known = create_preamble_ltf(FFT_size, CP_len)
    data_payload, _ = create_data_with_pilots(10, FFT_size, CP_len)
    
    # 儲存 LTF 用於 Rx 驗證
    np.savez('ltf_data.npz', ltf_freq_known=ltf_known)
    
    # 組合 Frame
    tx_waveform = np.hstack([preamble_sc, preamble_ltf, data_payload])
    
    # 功率正規化
    tx_waveform = tx_waveform / np.max(np.abs(tx_waveform)) * 0.5
    tx_waveform = tx_waveform.astype(np.complex64).reshape(1, -1)
    
    # 2. 設定 USRP
    dev_addr = f"serial={serial}" if serial else "type=b200"
    usrp_tx = uhd.usrp.MultiUSRP(dev_addr)
    usrp_tx.set_tx_rate(Fs)
    usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
    usrp_tx.set_tx_gain(TX_GAIN, 0)
    
    # 3. 發射
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [0]
    tx_streamer = usrp_tx.get_tx_stream(stream_args)
    metadata = uhd.types.TXMetadata()
    
    print(f"正在發送... (Gain: {TX_GAIN} dB)")
    
    try:
        while True:
            tx_streamer.send(tx_waveform, metadata)
    except KeyboardInterrupt:
        print("停止發送")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default=SERIAL_TX)
    args = parser.parse_args()
    transmit(args.serial)
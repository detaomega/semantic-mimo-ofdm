import numpy as np

# --- 基礎參數 ---
CENTER_FREQ = 2.45e9
SAMPLE_RATE = 1e6
TX_GAIN = 60
RX_GAIN = 55

# --- OFDM 參數 ---
FFT_SIZE = 64
CP_LEN = 16
TOTAL_LEN = FFT_SIZE + CP_LEN

# --- 子載波配置 (關鍵修改) ---
# 全部有效子載波 (-26 到 26, 不含 DC)
ALL_ACTIVE_BINS = np.arange(-26, 27)
ALL_ACTIVE_BINS = ALL_ACTIVE_BINS[ALL_ACTIVE_BINS != 0]

# 指定 Pilot 位置 (模仿 WiFi: -21, -7, 7, 21)
PILOT_BINS = np.array([-21, -7, 7, 21])
PILOT_VALUE = 1.0 + 0j # Pilot 發送的已知值

# 指定 Data 位置 (從全部裡面扣掉 Pilot)
DATA_BINS = np.setdiff1d(ALL_ACTIVE_BINS, PILOT_BINS)

NUM_DATA_CARRIERS = len(DATA_BINS)

# --- 1. 產生 Schmidl & Cox Preamble [A, A] ---
# (這部分維持不變，用於粗略同步)
np.random.seed(1)
_pn = np.random.choice([-1, 1], FFT_SIZE // 2) + 1j * np.random.choice([-1, 1], FFT_SIZE // 2)
_preamble_freq = np.zeros(FFT_SIZE, dtype=np.complex64)
_preamble_freq[0::2] = _pn 
_time_preamble = np.fft.ifft(_preamble_freq) * np.sqrt(FFT_SIZE)
PREAMBLE_SYMBOL = np.concatenate([_time_preamble[:FFT_SIZE//2], _time_preamble[:FFT_SIZE//2]])
PREAMBLE_SYMBOL /= np.max(np.abs(PREAMBLE_SYMBOL))
KNOWN_PREAMBLE_FREQ = np.fft.fft(PREAMBLE_SYMBOL) / np.sqrt(FFT_SIZE)

# --- 2. 產生 OFDM-QPSK Data Symbol (含 Pilot) ---
# 隨機產生 QPSK 符號
np.random.seed(42)
_bits = np.random.randint(0, 2, (NUM_DATA_CARRIERS, 2))
QPSK_SYMBOLS = ((2*_bits[:,0]-1) + 1j*(2*_bits[:,1]-1)) / np.sqrt(2)

# 映射到頻率軸
_data_freq = np.zeros(FFT_SIZE, dtype=np.complex64)

# A. 填入 Data
_data_indices = (DATA_BINS + FFT_SIZE) % FFT_SIZE
_data_freq[_data_indices] = QPSK_SYMBOLS

# B. 填入 Pilot (關鍵！)
_pilot_indices = (PILOT_BINS + FFT_SIZE) % FFT_SIZE
_data_freq[_pilot_indices] = PILOT_VALUE

# IFFT 轉時域
_time_data = np.fft.ifft(_data_freq) * np.sqrt(FFT_SIZE)
DATA_SYMBOL = np.concatenate([_time_data[-CP_LEN:], _time_data])
DATA_SYMBOL /= np.max(np.abs(DATA_SYMBOL)) * 1.5 

# 完整的發送單元
TX_BLOCK = np.concatenate([PREAMBLE_SYMBOL, DATA_SYMBOL]).astype(np.complex64)
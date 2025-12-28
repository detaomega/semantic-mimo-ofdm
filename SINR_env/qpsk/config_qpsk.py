import numpy as np

# --- 基礎參數 ---
CENTER_FREQ = 2.45e9
SAMPLE_RATE = 1e6
TX_GAIN = 45   # 調整至甜蜜點
RX_GAIN = 30   # 調整至甜蜜點

# --- OFDM 參數 ---
FFT_SIZE = 64
CP_LEN = 16
TOTAL_LEN = FFT_SIZE + CP_LEN

# 子載波配置 (類似 WiFi 802.11a, 使用中間 52 個)
# 頻率索引: -26 到 26 (扣除 0)
OCCUPIED_BINS = np.arange(-26, 27)
OCCUPIED_BINS = OCCUPIED_BINS[OCCUPIED_BINS != 0]
NUM_DATA_CARRIERS = len(OCCUPIED_BINS)

# --- 1. 產生 Schmidl & Cox Preamble [A, A] ---
# 用於同步和粗略頻偏估計
np.random.seed(1)
_pn = np.random.choice([-1, 1], FFT_SIZE // 2) + 1j * np.random.choice([-1, 1], FFT_SIZE // 2)
# 間隔插值產生時域重複特性
_preamble_freq = np.zeros(FFT_SIZE, dtype=np.complex64)
_preamble_freq[0::2] = _pn 
# IFFT
_time_preamble = np.fft.ifft(_preamble_freq) * np.sqrt(FFT_SIZE)
# 建立 [A, A] 結構 (長度 = FFT_SIZE)
PREAMBLE_SYMBOL = np.concatenate([_time_preamble[:FFT_SIZE//2], _time_preamble[:FFT_SIZE//2]])
# 歸一化
PREAMBLE_SYMBOL /= np.max(np.abs(PREAMBLE_SYMBOL))

# 頻域已知的 Preamble (用於通道估測)
# 注意：因為我們在時域重複 A，這等同於頻域只有偶數載波有值
# 為了通道估測，我們需要知道這些值
KNOWN_PREAMBLE_FREQ = np.fft.fft(PREAMBLE_SYMBOL) / np.sqrt(FFT_SIZE)


# --- 2. 產生 OFDM-QPSK Data Symbol ---
# 隨機產生 QPSK 符號
np.random.seed(42)
_bits = np.random.randint(0, 2, (NUM_DATA_CARRIERS, 2))
# 對應到 1+j, -1+j ...
QPSK_SYMBOLS = ((2*_bits[:,0]-1) + 1j*(2*_bits[:,1]-1)) / np.sqrt(2)

# 映射到頻率軸
_data_freq = np.zeros(FFT_SIZE, dtype=np.complex64)
# 這裡要小心頻率移位 (fftshift 的逆操作)
# 將 [-26...26] 映射到 FFT 的 [0...63] 索引
_fft_indices = (OCCUPIED_BINS + FFT_SIZE) % FFT_SIZE
_data_freq[_fft_indices] = QPSK_SYMBOLS

# IFFT 轉時域
_time_data = np.fft.ifft(_data_freq) * np.sqrt(FFT_SIZE)
# 加 CP
DATA_SYMBOL = np.concatenate([_time_data[-CP_LEN:], _time_data])
# 歸一化
DATA_SYMBOL /= np.max(np.abs(DATA_SYMBOL)) * 1.5 # 稍微降一點避免 Data 比 Preamble 大太多

# 完整的發送單元
TX_BLOCK = np.concatenate([PREAMBLE_SYMBOL, DATA_SYMBOL]).astype(np.complex64)
import numpy as np

# --- 參數設定 ---
CENTER_FREQ = 2.45e9
SAMPLE_RATE = 1e6
TX_GAIN = 50           
RX_GAIN = 40

# --- OFDM 參數 ---
FFT_SIZE = 64          # 子載波數量
CP_LEN = 16            # 循環字首長度 (保護區間)
TOTAL_LEN = FFT_SIZE + CP_LEN

# 有效子載波 (佔用中間 52 個，左右留空作為保護頻帶)
# 這跟 802.11a/g WiFi 的規格很像
OCCUPIED_BINS = np.arange(-26, 27) 
OCCUPIED_BINS = OCCUPIED_BINS[OCCUPIED_BINS != 0] # 去掉中間的 DC (直流)

# --- 產生已知的頻域 Pilot ---
np.random.seed(42)
# 產生 QPSK 符號
num_pilots = len(OCCUPIED_BINS)
_bits = np.random.randint(0, 2, (num_pilots, 2))
qpsk_symbols = ((2*_bits[:,0]-1) + 1j*(2*_bits[:,1]-1)) / np.sqrt(2)

# 建立完整的頻域符號 (由 0 和 Pilot 組成)
FREQ_PILOTS = np.zeros(FFT_SIZE, dtype=np.complex64)
# 將 QPSK 塞入對應的子載波位置 (需做 fftshift 處理頻率順序)
indices = (OCCUPIED_BINS + FFT_SIZE) % FFT_SIZE
FREQ_PILOTS[indices] = qpsk_symbols

# --- 產生對應的時域波形 (用於發射與同步) ---
# 1. IFFT 轉到時域
time_signal = np.fft.ifft(FREQ_PILOTS) * np.sqrt(FFT_SIZE) # 能量正規化
# 2. 加上 CP (複製尾巴放到頭部)
TIME_SYMBOL = np.concatenate([time_signal[-CP_LEN:], time_signal])
TIME_SYMBOL = TIME_SYMBOL.astype(np.complex64)
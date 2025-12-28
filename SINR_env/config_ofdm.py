import numpy as np

CENTER_FREQ = 2.45e9
SAMPLE_RATE = 1e6
TX_GAIN = 45 # 預設
RX_GAIN = 30 # 預設

FFT_SIZE = 64
CP_LEN = 16

# --- Schmidl & Cox Preamble 製作 ---
# 關鍵：建立一個在時域上重複兩次的符號 [A, A]
# 我們用 PN Sequence 來產生 A
np.random.seed(123)
L = FFT_SIZE // 2 
pn = np.random.choice([-1, 1], L) + 1j * np.random.choice([-1, 1], L)
# IFFT 轉到時域
preamble_half = np.fft.ifft(pn)
# 重複兩次
SC_PREAMBLE = np.concatenate([preamble_half, preamble_half])
# 歸一化振幅
SC_PREAMBLE = SC_PREAMBLE / np.max(np.abs(SC_PREAMBLE)) * 0.8
SC_PREAMBLE = SC_PREAMBLE.astype(np.complex64)

# Data Pilot (用於 SINR 計算)
data_bits = np.random.randint(0, 2, (FFT_SIZE, 2))
qpsk = ((2*data_bits[:,0]-1) + 1j*(2*data_bits[:,1]-1))/np.sqrt(2)
time_data = np.fft.ifft(qpsk) * np.sqrt(FFT_SIZE)
DATA_SYMBOL = np.concatenate([time_data[-CP_LEN:], time_data]).astype(np.complex64)
KNOWN_QPSK = qpsk # 頻域真值
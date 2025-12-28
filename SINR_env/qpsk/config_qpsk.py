import numpy as np

# --- 基礎設定 ---
CENTER_FREQ = 2.45e9
SAMPLE_RATE = 1e6

# --- 增益 (請根據之前的經驗微調) ---
# 建議先設在中間值，避免飽和也避免太弱
TX_GAIN = 45
RX_GAIN = 30 

# --- 產生 QPSK 序列 ---
# 我們產生一段固定長度的 QPSK 作為測試樣本
BLOCK_LEN = 1000 
np.random.seed(999) # 固定種子

# 產生 0~3 的整數
_syms = np.random.randint(0, 4, BLOCK_LEN)

# 對應到複數平面 (1+j, -1+j, -1-j, 1-j)
# 這裡我們用 exp(j * (pi/4 + k*pi/2))
QPSK_SEQ = np.exp(1j * (np.pi/4 + _syms * np.pi/2)).astype(np.complex64)

# 為了方便同步，我們取前 100 個點當作「同步標頭 (Sync Header)」
SYNC_SEQ = QPSK_SEQ[:100]
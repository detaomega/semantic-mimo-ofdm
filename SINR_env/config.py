import numpy as np

# --- 參數設定 ---
CENTER_FREQ = 2.45e9   # 2.45 GHz
SAMPLE_RATE = 1e6      # 1 Msps
N_PILOTS = 40          # Pilot 長度

# --- 增益設定 (近距離優化版) ---
# Tx=50, Rx=40 是 B210 在室內近距離 (約 50cm - 1m) 的甜蜜點
TX_GAIN = 50           
RX_GAIN = 40           

# --- 產生 Pilot 序列 ---
np.random.seed(42) # 重要：固定種子確保序列一致

# 產生 QPSK 符號
_bits = np.random.randint(0, 2, (N_PILOTS, 2))
# 歸一化 (Normalization)
KNOWN_PILOTS = ((2*_bits[:,0]-1) + 1j*(2*_bits[:,1]-1)) / np.sqrt(2)
KNOWN_PILOTS = KNOWN_PILOTS.astype(np.complex64)
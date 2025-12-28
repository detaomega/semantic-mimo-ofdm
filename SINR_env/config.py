import numpy as np

# --- 參數設定 ---
CENTER_FREQ = 2.45e9   # 2.45 GHz
SAMPLE_RATE = 1e6      # 1 Msps
N_PILOTS = 40          # Pilot 長度 (參照文件 N=40) [cite: 207]

# --- 關鍵修正：增益設定 ---
# 之前的 70/65 太強會導致 Clipping，這裡改為較安全的數值
# 如果距離拉遠到 3 公尺以上，可以再適度調高 Rx Gain
TX_GAIN = 45           
RX_GAIN = 40           

# --- 產生 Pilot 序列 ---
# 固定 Seed，確保 Tx 和 Rx 產生一模一樣的序列
np.random.seed(42)

# 產生 QPSK 符號
_bits = np.random.randint(0, 2, (N_PILOTS, 2))
# 歸一化 (Normalization) 使功率為 1
KNOWN_PILOTS = ((2*_bits[:,0]-1) + 1j*(2*_bits[:,1]-1)) / np.sqrt(2)
KNOWN_PILOTS = KNOWN_PILOTS.astype(np.complex64)
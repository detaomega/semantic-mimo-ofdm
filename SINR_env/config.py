import numpy as np


CENTER_FREQ = 2.45e9   # 中心頻率 2.45 GHz (避開 WiFi干擾較嚴重的 Ch1/6/11)
SAMPLE_RATE = 1e6      # 取樣率 1 Msps
N_PILOTS = 40          # Pilot 長度，參照文件設定 [cite: 207]
TX_GAIN = 40           # 發射增益 (0 - 89.8 dB)
RX_GAIN = 30           # 接收增益 (0 - 76 dB)

np.random.seed(42)
_bits = np.random.randint(0, 2, (N_PILOTS, 2))

KNOWN_PILOTS = ((2*_bits[:,0]-1) + 1j*(2*_bits[:,1]-1)) / np.sqrt(2)
KNOWN_PILOTS = KNOWN_PILOTS.astype(np.complex64)
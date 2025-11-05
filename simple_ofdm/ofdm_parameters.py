import numpy as np

# --- 1. 核心參數 ---
FFT_SIZE = 64
FS = 10e6           # 取樣率 (10 MHz)
FC = 0.915e9        # 載波頻率 (915 MHz)
M = 16              # QAM 階數 (16-QAM)
NUM_SYMBOLS = 100     # OFDM 資料符號數量

# --- 2. 訓練序列 (Preamble) 定義 (頻域) ---
# 短訓練序列 (STS)
S = np.sqrt(13/6) * np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j,
    0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j,
    0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0
], dtype=np.complex64)

# 長訓練序列 (LTS)
L = np.array([
    0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1,
    1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1,
    -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0
], dtype=np.complex64)

# --- 3. 子載波索引 (Python 0-based) ---
DATA_INDICES = np.concatenate([
    np.arange(6, 11),     # 7:11  (5)
    np.arange(12, 25),    # 13:25 (13)
    np.arange(26, 32),    # 27:32 (6)
    np.arange(33, 39),    # 34:39 (6)
    np.arange(40, 53),    # 41:53 (13)
    np.arange(54, 59)     # 55:59 (5)
]) # 總共 48 個資料子載波

PILOT_INDICES = np.array([11, 25, 39, 53])
PILOT_TONES = np.array([1+0j, 1+0j, -1+0j, -1+0j], dtype=np.complex64)

REMOVE_INDICES = np.concatenate([
    np.arange(0, 6),   # 0-5
    np.array([11, 25, 32, 39, 53]), # Pilots + DC
    np.arange(59, 64)  # 59-63
])

# --- 4. 16-QAM 調變/解調 ---
LOG2_M = int(np.log2(M))
QAM_MAP = (1/np.sqrt(10)) * np.array([
    -3-3j, -3-1j, -3+3j, -3+1j,
    -1-3j, -1-1j, -1+3j, -1+1j,
     3-3j,  3-1j,  3+3j,  3+1j,
     1-3j,  1-1j,  1+3j,  1+1j
], dtype=np.complex64)

def map_16qam(bits):
    num_bits = len(bits)
    assert num_bits % LOG2_M == 0, "Bits length must be a multiple of 4"
    indices = bits.reshape((num_bits // LOG2_M, LOG2_M))
    # 轉換 4 bits (e.g., [1, 0, 1, 0]) 為 0-15 整數 (e.g., 10)
    bit_indices = (indices[:, 0] * 8 + indices[:, 1] * 4 + indices[:, 2] * 2 + indices[:, 3]).astype(int)
    return QAM_MAP[bit_indices]

def demap_16qam(symbols):
    symbols_flat = symbols.flatten()
    bits = np.zeros((len(symbols_flat), LOG2_M), dtype=np.uint8)
    for i, sym in enumerate(symbols_flat):
        distances = np.abs(sym - QAM_MAP)
        index = np.argmin(distances)
        bits[i, 0] = (index >> 3) & 1
        bits[i, 1] = (index >> 2) & 1
        bits[i, 2] = (index >> 1) & 1
        bits[i, 3] = (index >> 0) & 1
    return bits.flatten()

# --- 5. Preamble (時域) 產生函數 ---
def generate_preambles():
    STS = np.fft.ifft(np.fft.fftshift(S)) * FFT_SIZE
    cp_size_sts = FFT_SIZE // 2  # 32
    sts_with_cp = np.concatenate([STS[FFT_SIZE - cp_size_sts:], STS, STS]) # 160
    LTS = np.fft.ifft(np.fft.fftshift(L)) * FFT_SIZE
    cp_size_lts = FFT_SIZE // 2  # 32
    lts_with_cp = np.concatenate([LTS[FFT_SIZE - cp_size_lts:], LTS, LTS]) # 160
    return sts_with_cp, lts_with_cp

# --- 6. 完整封包長度 (用於接收) ---
START_ZERO_LEN = 50
END_ZERO_LEN = 130
CP_SIZE_DATA = 16
SYMBOL_LEN_DATA = FFT_SIZE + CP_SIZE_DATA # 80
DATA_PAYLOAD_LEN = NUM_SYMBOLS * SYMBOL_LEN_DATA # 8000
STS_LEN = 160
LTS_LEN = 160
FRAME_LEN = START_ZERO_LEN + STS_LEN + LTS_LEN + DATA_PAYLOAD_LEN + END_ZERO_LEN
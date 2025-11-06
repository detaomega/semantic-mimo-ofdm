import numpy as np

# --- 檔案名稱 (固定) ---
GT_FILE_LTF = 'ltf_data.npz'
GT_FILE_PAYLOAD = 'gt_simple.npz'

# --- OFDM 參數 (必須與 Tx/Rx 一致) ---
FFT_size = 64
num_data_carriers = 48
num_data_symbols = 10

# --- 1. 產生 LTF (通道估測用) ---
print(f"Generating LTF (for Channel Estimation)...")
rng = np.random.RandomState(seed=123) # 固定種子，確保可重複
ltf_freq_known = rng.choice([-1, 1], FFT_size)
np.savez(GT_FILE_LTF, ltf_freq_known=ltf_freq_known)
print(f"Saved LTF data to {GT_FILE_LTF}")

# --- 2. 產生「任意」 Complex Number (Ground Truth) ---
print(f"Generating ARBITRARY complex payload (Ground Truth)...")

# 隨機產生 I 和 Q 值 (範圍在 -1.0 到 +1.0 之間)
I_values = (np.random.rand(num_data_symbols, num_data_carriers) - 0.5) * 2.0
Q_values = (np.random.rand(num_data_symbols, num_data_carriers) - 0.5) * 2.0

# 這就是我們的「任意 complex number」
arbitrary_payload_freq = (I_values + 1j*Q_values).astype(np.complex64)

# 範例：檢查一個值 (可能就像 0.4 + 0.3j)
print(f"Example arbitrary complex number: {arbitrary_payload_freq[0,0]}")

# 儲存 Ground Truth
np.savez(GT_FILE_PAYLOAD, gt_payload_freq=arbitrary_payload_freq)
print(f"Saved Arbitrary Ground Truth payload to {GT_FILE_PAYLOAD}")
print("\nDone. 你現在可以執行 Tx 和 Rx 了。")
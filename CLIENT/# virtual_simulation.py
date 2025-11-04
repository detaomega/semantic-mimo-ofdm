# virtual_simulation_sweep.py
#
# 1. 讀取 Ground Truth Payload (gt_data.npz)
# 2. [執行一次] 模擬 OFDM 發射
# 3. [迴圈] 針對一系列 SNR 數值：
#    a. 模擬 AWGN 通道 (加入隨機雜訊)
#    b. 模擬 OFDM 接收
#    c. 計算 MSE 和 ESNR
# 4. 繪製效能曲線圖

import numpy as np
import matplotlib.pyplot as plt
from ofdm_cdh import OFDM_FrameGenerator
from tqdm import tqdm # 匯入 tqdm 來顯示進度條

# --- 模擬參數 ---
GT_FILE = 'gt_data.npz'

# --- 這是您要測試的 SNR 範圍 ---
SNR_dB_LIST = np.arange(0.0, 40.1, 2.0) # 從 0 dB 到 30 dB，每 2 dB 測試一次
# -------------------------------

# --- OFDM 參數 (必須與 Tx/Rx 腳本完全一致) ---
Fs = 15 * 128 * 1000
N = 28.0 
PILOT_NORM = 4.0

# --- 1. [執行一次] 初始化 OFDM Mapper (1x1 SISO) ---
print("Initializing OFDM mapper...")
mapper = OFDM_FrameGenerator(
    num_subcarriers=72, DC_guard=1, slots_per_frame=50, symbols_per_slot=7,
    pilot_place=0, sync_place=0, subcarrier_spacing=15*1000, FFT_size=128,
    num_cp_samples=9, num_ex_cp_samples=10, sequential_mapping=True,
    initial_pad=True, num_antenna=1, antenna_idx=0
)
N_SAMPLES_PER_FRAME = mapper.n_samples_per_frame # 48000

# --- 2. [執行一次] 讀取 Ground Truth 檔案 ---
print(f"Loading Ground Truth from {GT_FILE}...")
try:
    gt_data = np.load(GT_FILE)
    gt_payload = gt_data['gt_payload'] 
    payload_length = gt_payload.shape[0] 
except FileNotFoundError:
    print(f"!!! 錯誤: {GT_FILE} 找不到 !!!")
    print("請先執行 'generate_payload.py'。")
    exit()
print(f"Ground Truth loaded ({payload_length} symbols).")

# --- 3. [執行一次] 計算 Ground Truth 功率 ---
gt_sym_pow = np.mean(np.abs(gt_payload)**2)

# --- 4. [執行一次] 模擬發射端 (Tx) ---
print("Simulating Transmitter (Tx) waveform (once)...")
payload_norm = gt_payload / N 
payload_tx = payload_norm.reshape(-1, 1).T 

symbol_frame = mapper.mapToFrame(payload_tx[0]) 
tx_waveform = mapper.symbolsToSignal(symbol_frame).signal 

tx_waveform_scaled = tx_waveform.copy()
tx_waveform_scaled[822:960] /= 1.7
tx_waveform_scaled = tx_waveform_scaled.reshape(-1, 960)
tx_waveform_scaled[:,:138] /= (PILOT_NORM / np.sqrt(1))
tx_waveform_scaled = tx_waveform_scaled.flatten()

# --- 5. [執行一次] 計算發射訊號功率 ---
tx_signal_power = np.mean(np.abs(tx_waveform_scaled)**2)
print("Tx simulation complete. Starting SNR sweep...")

# --- 6. [迴圈] 針對每個 SNR 進行模擬 ---
results_esnr = []
results_mse = []

# 使用 tqdm 顯示進度條
for snr_db in tqdm(SNR_dB_LIST, desc="Running SNR Sweep"):
    
    # --- 6a. 模擬 AWGN 通道 ---
    # 從 SNR 計算雜訊功率
    snr_linear = 10**(snr_db / 10.0)
    noise_variance = tx_signal_power / snr_linear
    
    # 產生隨機雜訊 (Complex Gaussian Noise)
    noise_std_dev = np.sqrt(noise_variance / 2.0)
    noise = np.random.normal(0, noise_std_dev, size=tx_waveform_scaled.shape) + \
         1j * np.random.normal(0, noise_std_dev, size=tx_waveform_scaled.shape)
    
    # 加入雜訊
    rx_waveform = tx_waveform_scaled + noise

    # --- 6b. 模擬接收端 (Rx) ---
    rcv_waveform = rx_waveform[0 : N_SAMPLES_PER_FRAME] # 完美同步
    
    rcv_waveform_copy = rcv_waveform.copy()
    rcv_waveform_copy[822:960] *= 1.7
    rcv_waveform_copy = rcv_waveform_copy.reshape(-1, 960)
    rcv_waveform_copy[:,:138] *= (PILOT_NORM / np.sqrt(1))
    rcv_waveform_copy = rcv_waveform_copy.flatten()

    rcv_symbol = mapper.signalToSymbols(rcv_waveform_copy)
    
    channels = mapper.get_mimo_channel(rcv_symbol)
    channels = np.stack([channels], axis=1)
    
    rcv_symbol = rcv_symbol.reshape(rcv_symbol.shape[0], 1, rcv_symbol.shape[1])
    rcv_symbols_zf = mapper.mimo_zf_equalize(rcv_symbol, channels)
    
    rcv_payload = mapper.extractPayloads(rcv_symbols_zf[0])
    rcv_payload = rcv_payload[:payload_length].astype(np.complex64)
    
    rcv_payload *= N # 反正規化

    # --- 6c. 計算 MSE 和 ESNR ---
    mse = np.mean(np.abs(gt_payload - rcv_payload)**2)
    esnr = 10*np.log10(gt_sym_pow / mse)
    
    results_mse.append(mse)
    results_esnr.append(esnr)

print("Simulation sweep complete.")

# --- 7. 繪製結果圖表 ---
print("Plotting results...")
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
fig.suptitle('OFDM Simulation Performance (AWGN Channel)', fontsize=16)

# 圖 1: ESNR vs. SNR
ax1.plot(SNR_dB_LIST, results_esnr, 'bo-', label='Measured Payload ESNR')
ax1.plot(SNR_dB_LIST, SNR_dB_LIST, 'r--', label='Ideal (ESNR = SNR)') # 理想曲線
ax1.set_xlabel('AWGN Channel SNR (dB)')
ax1.set_ylabel('Payload ESNR (dB)')
ax1.set_title('ESNR vs. Channel SNR')
ax1.legend()
ax1.grid(True)

# 圖 2: MSE vs. SNR
ax2.plot(SNR_dB_LIST, results_mse, 'go-')
ax2.set_xlabel('AWGN Channel SNR (dB)')
ax2.set_ylabel('Payload MSE (log scale)')
ax2.set_title('Mean Squared Error (MSE) vs. Channel SNR')
ax2.set_yscale('log') # MSE 通常用 log scale 來看
ax2.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('virtual_simulation_snr_sweep.png')
plt.show()

print("All done. Check 'virtual_simulation_snr_sweep.png'")
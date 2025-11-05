import numpy as np
import uhd
import time

from utils import interleave
from ofdm_cdh import OFDM_FrameGenerator

SERIAL_TX = "3475843"       # 您的 B210 序列號
TX_GAIN = 15.0              # 發射增益 (dB)
GT_FILE = 'gt_data.npz'     # 要讀取的 Ground Truth 檔案

Fs = 15 * 128 * 1000
Fc = 5.1e9                  # <<<--- 關鍵修改：必須與 Rx 相同
N = 28.0 # Payload normalization
# PILOT_NORM = 4.0 # 已移至 ofdm_cdh.py 內部

# --- 1. Initial OFDM Mapper (1x1 SISO) ---
print("Initializing OFDM mapper...")
mapper = OFDM_FrameGenerator(
    num_subcarriers=72, DC_guard=1, slots_per_frame=50, symbols_per_slot=7,
    pilot_place=0, sync_place=0, subcarrier_spacing=15*1000, FFT_size=128,
    num_cp_samples=9, num_ex_cp_samples=10, sequential_mapping=True,
    initial_pad=True, num_antenna=1, antenna_idx=0
)

# --- 2. Loading Ground Truth ---
print(f"Loading Ground Truth from {GT_FILE}...")
try:
    gt_data = np.load(GT_FILE)
    payload = gt_data['gt_payload'] # 讀取 payload
    PAYLOAD_LEN = payload.shape[0]
except FileNotFoundError:
    print(f"!!! 錯誤: {GT_FILE} 找不到 !!!")
    print("請先執行 'generate_payload.py'，並將 'gt_data.npz' 複製到這個資料夾。")
    exit()
print(f"Ground truth loaded ({PAYLOAD_LEN} symbols).")


# --- 3. Preparing waveform for transmission ---
print("Preparing waveform for transmission...")
payload_norm = payload / N # 正規化
payload_tx = payload_norm.reshape(-1, 1).T # shape (1, 512)

symbol_frame = mapper.mapToFrame(payload_tx[0]) 

# symbolsToSignal 現在會自動處理 IFFT, CP 和功率調整
waveform_obj = mapper.symbolsToSignal(symbol_frame)
waveform = waveform_obj.signal 

# (已移除) "Adjust" 區塊 - 功率調整邏輯已移至 mapper.symbolsToSignal 內部

# Final Sending Sample
tx_waveform = waveform.reshape(1, -1).astype(np.complex64)
print(f"Waveform ready (shape: {tx_waveform.shape})")

# --- 4. Connecting to TX USRP (B210) ---
print(f"Connecting to TX USRP (B210) at serial={SERIAL_TX}...")
usrp_tx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_TX}"))

# --- 5. Setting B210 ---
usrp_tx.set_clock_source("internal", 0)
usrp_tx.set_time_source("internal", 0)
usrp_tx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

usrp_tx.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:A"), 0)
usrp_tx.set_tx_antenna("TX/RX", 0)
usrp_tx.set_tx_rate(Fs)
usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_tx.set_tx_gain(TX_GAIN, 0)
print(f"B210 (Tx) setup complete. Rate: {Fs/1e6} MHz, Freq: {Fc/1e9} GHz, Gain: {TX_GAIN} dB")

# --- 6. Sending stream ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
tx_streamer = usrp_tx.get_tx_stream(stream_args)

# --- 7. Starting continuous transmission ---
print("\n*** Starting continuous transmission... (Press Ctrl+C to stop) ***")
tx_metadata = uhd.types.TXMetadata()
tx_metadata.start_of_burst = True
tx_metadata.end_of_burst = False
tx_metadata.has_time_spec = True

next_tx_time = usrp_tx.get_time_now().get_real_secs() + 0.2
tx_metadata.time_spec = uhd.types.TimeSpec(next_tx_time)
tx_streamer.send(tx_waveform, tx_metadata)

try:
    while True:
        next_tx_time += 0.025
        tx_metadata.time_spec = uhd.types.TimeSpec(next_tx_time)
        tx_streamer.send(tx_waveform, tx_metadata)
        
except KeyboardInterrupt:
    print("\nStopping transmission...")
    tx_metadata.end_of_burst = True
    tx_streamer.send(np.zeros_like(tx_waveform), tx_metadata)
    print("Transmitter shut down.")
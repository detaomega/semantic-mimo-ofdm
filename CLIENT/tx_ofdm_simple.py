import numpy as np
import uhd
import time

from ofdm_cdh import OFDM_FrameGenerator
from tcp_configs import USRP_ADDR0 # 只需要 USRP 位址

# --- 設定 ---
Fs = 15 * 128 * 1000
Fc = 5.4e9
Tx_gain = 60
pilot_norm = 4.0
payload_norm = 28.0
num_Tx = 1
SERIAL_TX = '3475843' # 你的 Tx 序列號 (請確認)
GT_FILE = "gt_payload_local.npz"

# --- 1. 初始化 OFDM Mapper (SISO) ---
print("Initializing OFDM mapper for SISO Tx...")
mapper = OFDM_FrameGenerator(
    num_subcarriers=72, DC_guard=1, slots_per_frame=50, symbols_per_slot=7,
    pilot_place=0, sync_place=0, subcarrier_spacing=15*1000, FFT_size=128,
    num_cp_samples=9, num_ex_cp_samples=10, sequential_mapping=True,
    initial_pad=True, 
    num_antenna=1,
    antenna_idx=0
)
PAYLOAD_LEN = mapper.num_data
print(f"Total data carriers: {PAYLOAD_LEN}")

# --- 2. 連接 USRP (發射器) ---
print(f"Connecting to TX USRP at {SERIAL_TX}...")
usrp_tx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_TX}"))

# (關鍵) 使用內部時脈
usrp_tx.set_clock_source("internal")
usrp_tx.set_time_source("internal")
usrp_tx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

# 設定 1 個 Tx 通道
usrp_tx.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:A"), 0)
usrp_tx.set_tx_antenna("TX/RX", 0)
usrp_tx.set_tx_rate(Fs)
usrp_tx.set_tx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_tx.set_tx_gain(Tx_gain, 0)

print(f"Tx Rate: {usrp_tx.get_tx_rate()/1e6} MHz")
print(f"Tx Gain Chan 0: {usrp_tx.get_tx_gain(0)} dB")

# 建立 1 通道的 Tx Streamer
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
tx_streamer = usrp_tx.get_tx_stream(stream_args)
tx_metadata = uhd.types.TXMetadata()
tx_metadata.start_of_burst = False
tx_metadata.end_of_burst = False
tx_metadata.has_time_spec = False

# --- 3. (修改) 本地隨機產生 Payload ---
print("Generating random QPSK payload...")
# 產生 QPSK 星座點 (標準正規化)
qpsk_constellation = [1+1j, 1-1j, -1+1j, -1-1j] / np.sqrt(2)
# 隨機選取 PAYLOAD_LEN 個 QPSK 符元
gt_payload = np.random.choice(qpsk_constellation, PAYLOAD_LEN).astype(np.complex64)

# (關鍵) 儲存 Ground Truth 檔案，讓 Rx 可以讀取
np.savez(GT_FILE, gt_payload=gt_payload)
print(f"Ground truth payload saved to {GT_FILE}")

# --- 4. 準備 OFDM 波形 (僅執行一次) ---
print("Preparing OFDM waveform...")
payload = gt_payload / payload_norm # 使用你的正規化
symbol_frame = mapper.mapToFrame(payload)
waveform = mapper.symbolsToSignal(symbol_frame).signal

# 套用你的 "magic number" 功率調整
if pilot_norm != 1:
    waveform[822:960] /= 1.7
    waveform = waveform.reshape(-1, 960)
    waveform[:,:138] /= (pilot_norm / np.sqrt(1)) 
    waveform = waveform.flatten()
    
# 最終的 1 通道波形 (shape 必須是 (1, N) )
waveform_siso = waveform.astype(np.complex64).reshape(1, -1)
print(f"Waveform ready. Shape: {waveform_siso.shape}")

# --- 5. 主發射迴圈 ---
print("\n*** Starting continuous transmission... (Press Ctrl+C to stop) ***")
try:
    while True:
        # 連續發射同一個波形
        tx_streamer.send(waveform_siso, tx_metadata)
        
except KeyboardInterrupt:
    print("\nStopping transmission...")

finally:
    # 關閉串流
    tx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    print("Tx shut down.")
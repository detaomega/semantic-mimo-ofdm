# rx_client.py
#
# 1. 從 gt_data.npz 讀取 Ground Truth Payload
# 2. 連接到 B200mini (Rx)
# 3. 連續接收、同步、解碼
# 4. 計算 ESNR 並繪圖

import numpy as np
import uhd
import matplotlib.pyplot as plt
# (移除了 from IPython import display)
import time

from utils import interleave
from ofdm_cdh import OFDM_FrameGenerator

# --- 接收器設定 ---
SERIAL_RX = "34B733A"       # 您的 B200mini 序列號
RX_GAIN = 60.0              # 接收增益 (dB)
GT_FILE = 'gt_data.npz'     # Ground Truth 檔案
# --------------------

# --- OFDM 參數 ---
Fs = 15 * 128 * 1000
Fc = 5.1e9 # 必須與 Tx 相同
N = 28.0 # Payload normalization (必須與 Tx 相同)
# PILOT_NORM = 4.0 # 已移至 ofdm_cdh.py 內部

# --- 1. 初始化 OFDM Mapper (1x1 SISO) ---
print("Initializing OFDM mapper...")
mapper = OFDM_FrameGenerator(
    num_subcarriers=72, DC_guard=1, slots_per_frame=50, symbols_per_slot=7,
    pilot_place=0, sync_place=0, subcarrier_spacing=15*1000, FFT_size=128,
    num_cp_samples=9, num_ex_cp_samples=10, sequential_mapping=True,
    initial_pad=True, num_antenna=1, antenna_idx=0
)
N_SAMPLES_PER_FRAME = mapper.n_samples_per_frame # 48000

# --- 2. 讀取 Ground Truth 檔案 ---
print(f"Loading Ground Truth from {GT_FILE}...")
try:
    gt_data = np.load(GT_FILE)
    gt_payload = gt_data['gt_payload'] # 只讀取 payload
    payload_length = gt_payload.shape[0] # 512
except FileNotFoundError:
    print(f"!!! 錯誤: {GT_FILE} 找不到 !!!")
    print("請先在 Tx 主機上執行 tx_client.py，然後將 gt_data.npz 複製到這裡。")
    exit()
print(f"Ground Truth loaded ({payload_length} symbols).")

# --- 3. 連接到 B200mini (Rx) ---
print(f"Connecting to RX USRP (B200mini) at serial={SERIAL_RX}...")
usrp_rx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_RX}"))

# --- 4. 設定 B200mini ---
usrp_rx.set_clock_source("internal", 0)
usrp_rx.set_time_source("internal", 0)
usrp_rx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

usrp_rx.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A"), 0) # B200mini 使用 A:0
usrp_rx.set_rx_antenna("RX2", 0)
usrp_rx.set_rx_rate(Fs)
usrp_rx.set_rx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_rx.set_rx_gain(RX_GAIN, 0)
print(f"B200mini (Rx) setup complete. Rate: {Fs/1e6} MHz, Freq: {Fc/1e9} GHz, Gain: {RX_GAIN} dB")

# --- 5. 建立接收串流 ---
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
rx_streamer = usrp_rx.get_rx_stream(stream_args)

num_samps_to_recv = int(N_SAMPLES_PER_FRAME * 2.5) 
recv_buffer = np.zeros(num_samps_to_recv, dtype=np.complex64)
print(f"Receiver buffer created (size: {num_samps_to_recv})")

# --- 6. 準備繪圖 ---
print("Setting up live plot...")
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(18, 5))
fig.tight_layout(pad=4.0)

# --- 7. 連續接收與處理 ---
print("\n*** Starting continuous reception... (Press Ctrl+C to stop) ***")
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
rx_streamer.issue_stream_cmd(stream_cmd)
metadata = uhd.types.RXMetadata()

frame_counter = 0 
PLOT_EVERY_N_FRAMES = 30
try:
    while True:
        # 1. 接收 Smaples
        num_rx_samps = rx_streamer.recv(recv_buffer, metadata)
        
        if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
            print("O", end="", flush=True) 
            continue 
        elif metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f"Receiver Error: {metadata.strerror()}")
            continue
            
        # 2. 同步
        est_idx = mapper.synchronize(recv_buffer)
        
        if est_idx < 0 or (est_idx + N_SAMPLES_PER_FRAME) > num_samps_to_recv:
            print("S", end="", flush=True) # "S" for Sync failed
            continue 

        # 3. 擷取訊框
        rcv_waveform = recv_buffer[est_idx : est_idx + N_SAMPLES_PER_FRAME]
        
        # 4. 功率調整 (已移除 - 移至 OFDM_FrameGenerator.signalToSymbols 內部)
        
        # 5. FFT
        # signalToSymbols 現在會自動處理功率調整
        rcv_symbol = mapper.signalToSymbols(rcv_waveform)
        
        # 6. 通道估測與均衡
        # 使用 SISO 的 equalize 方法
        channels_for_plot = mapper.get_mimo_channel(rcv_symbol) # 僅用於繪圖
        rcv_symbols_eq = mapper.equalize(rcv_symbol) # 實際的均衡
        
        # 7. 擷取 Payload
        rcv_payload = mapper.extractPayloads(rcv_symbols_eq)
        rcv_payload = rcv_payload[:payload_length].astype(np.complex64)
        rcv_payload *= N 

        # 8. 計算 ESNR
        sym_pow = np.mean(np.abs(gt_payload)**2)
        err_pow = np.mean(np.abs(gt_payload - rcv_payload)**2)
        esnr = 10*np.log10(sym_pow / err_pow)
        
        frame_counter += 1
        
        # 9. 只有在需要時才繪圖
        if frame_counter % PLOT_EVERY_N_FRAMES == 0:
            print(f" [Plotting Update: ESNR {esnr:.2f} dB] ", end="", flush=True)
            
            # --- 圖 1: 通道 (ax1) ---
            ax1.clear()
            # 繪製通道的振幅響應 (對 pilot 取平均)
            channel_mag = np.abs(np.mean(channels_for_plot, axis=-1)[0])
            ax1.plot(channel_mag)
            ax1.set_xlim([0, 72])
            ax1.set_ylim(bottom=0)
            ax1.set_xlabel('Subcarrier Index')
            ax1.set_ylabel('Channel Magnitude')
            ax1.set_title('SISO Channel Magnitude')
            
            # --- 圖 2: PSD (ax2) ---
            ax2.clear()
            ax2.psd(recv_buffer, NFFT=1024, Fs=Fs, scale_by_freq=False, linewidth=0.1)
            ax2.set_title('Received Signal PSD')

            # ==================== MODIFIED BLOCK ====================
            # --- 圖 3: 星座圖 (ax3) ---
            ax3.clear()
            
            # 1. 繪製接收到的訊號 (藍色點雲)
            ax3.scatter(np.real(rcv_payload), np.imag(rcv_payload), 
                        s=0.5, alpha=0.3, label='Received (Rx)')
            
            # 2. 疊加繪製 Ground Truth (橘色點)
            ax3.scatter(np.real(gt_payload), np.imag(gt_payload), 
                        s=2, alpha=0.8, color='orange', label='Ground Truth (GT)')

            ax3.set_xlim([-1.5, 1.5]) 
            ax3.set_ylim([-1.5, 1.5])
            ax3.set_xlabel('In-Phase')
            ax3.set_ylabel('Quadrature-Phase')
            ax3.set_title(f'Constellation | ESNR:{esnr:.2f} dB, MSE: {err_pow:.4f}')
            ax3.legend() # <-- 顯示圖例
            ax3.grid(True)
            ax3.set_aspect('equal') # 保持 I/Q 軸 1:1
            # ======================================================
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

except KeyboardInterrupt:
    print("\nStopping reception...")
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd)
    plt.close()
    plt.ioff()
    print("Receiver shut down.")
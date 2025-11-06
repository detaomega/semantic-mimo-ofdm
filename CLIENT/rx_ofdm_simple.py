import numpy as np
import uhd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from ofdm_cdh import OFDM_FrameGenerator
SERIAL_RX = '34B733A'
GT_FILE = "gt_payload_local.npz"

# --- 設定 ---
Fs = 15 * 128 * 1000
Fc = 2.0e9
Rx_gain = 24.0
pilot_norm = 4.0
payload_norm = 28.0
num_Rx = 1

# --- 1. 初始化 OFDM Mapper (SISO) ---
print("Initializing OFDM mapper for SISO Rx...")
mapper = OFDM_FrameGenerator(
    num_subcarriers=72, DC_guard=1, slots_per_frame=50, symbols_per_slot=7,
    pilot_place=0, sync_place=0, subcarrier_spacing=15*1000, FFT_size=128,
    num_cp_samples=9, num_ex_cp_samples=10, sequential_mapping=True,
    initial_pad=True, 
    num_antenna=1,
    antenna_idx=0
)
N_SAMPLES_PER_FRAME = mapper.n_samples_per_frame
PAYLOAD_LEN = mapper.num_data

# --- 2. 連接 USRP (接收器) ---
print(f"Connecting to RX USRP at {SERIAL_RX}...")
usrp_rx = uhd.usrp.MultiUSRP(uhd.types.DeviceAddr(f"serial={SERIAL_RX}"))

usrp_rx.set_clock_source("internal")
usrp_rx.set_time_source("internal")
usrp_rx.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

usrp_rx.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A"), 0)
usrp_rx.set_rx_antenna("TX/RX", 0) 
usrp_rx.set_rx_rate(Fs)
usrp_rx.set_rx_freq(uhd.libpyuhd.types.tune_request(Fc), 0)
usrp_rx.set_rx_gain(Rx_gain, 0)
print(f"Rx Rate: {usrp_rx.get_rx_rate()/1e6} MHz, Rx Gain: {usrp_rx.get_rx_gain(0)} dB")

stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
stream_args.channels = [0]
rx_streamer = usrp_rx.get_rx_stream(stream_args)
metadata = uhd.types.RXMetadata()

# --- 3. 讀取本地 Ground Truth ---
try:
    gt_data = np.load(GT_FILE)
    gt_payload = gt_data['gt_payload']
    if len(gt_payload) != PAYLOAD_LEN:
        print(f"!!! 錯誤: Ground Truth 檔案長度 ({len(gt_payload)}) 與 Mapper ({PAYLOAD_LEN}) 不符 !!!")
        exit()
    print(f"Ground truth payload loaded from {GT_FILE}")
except FileNotFoundError:
    print(f"!!! 錯誤: 找不到 Ground Truth 檔案: {GT_FILE} !!!")
    print("請先執行 tx_siso_ofdm_local.py 來產生此檔案。")
    exit()

# --- 4. 準備繪圖 ---
print("Setting up live plot...")
plt.ion() # 開啟互動模式
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(18, 5))
fig.tight_layout(pad=4.0)

# --- 5. 主接收迴圈 (離線/突發模式) ---
print("\n*** Starting burst reception loop... (Press Ctrl+C to stop) ***")
try:
    while True:
        # --- A. 接收一個數據突發 (Burst) ---
        num_samps_to_recv = int(Fs * 0.5)
        recv_buffer = np.zeros(num_samps_to_recv, dtype=np.complex64)
        temp_buffer = np.zeros(10000, dtype=np.complex64)
        
        stream_cmd_start = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd_start.stream_now = True
        rx_streamer.issue_stream_cmd(stream_cmd_start)
        
        total_rx_samps = 0
        print(f"Receiving {num_samps_to_recv} samples burst...")
        while total_rx_samps < num_samps_to_recv:
            samps = rx_streamer.recv(temp_buffer, metadata)
            if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                print("O", end="", flush=True) 
                continue
            elif metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"Rx Error: {metadata.strerror()}")
            
            samps_to_copy = min(samps, num_samps_to_recv - total_rx_samps)
            if samps_to_copy <= 0:
                break
            
            recv_buffer[total_rx_samps : total_rx_samps + samps_to_copy] = temp_buffer[:samps_to_copy]
            total_rx_samps += samps_to_copy
        
        stream_cmd_stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(stream_cmd_stop)
        
        print("\nBurst reception complete. Processing...")
        rcv_waveform_full = recv_buffer

        # --- B. 離線處理 ---
        
        # 1. 同步 (SISO)
        rcv_waveform_raw = rcv_waveform_full
        est_idx = mapper.synchronize(rcv_waveform_raw)
        
        if est_idx < 0 or (est_idx + N_SAMPLES_PER_FRAME) > num_samps_to_recv:
            print(f"Sync failed! (est_idx: {est_idx}). Flushing buffer...")
            continue

        print(f"Sync success! Frame found at index {est_idx}.")

        # 2. Rx 通道估測 (SISO)
        rcv_waveform = rcv_waveform_raw[est_idx : est_idx + N_SAMPLES_PER_FRAME]
        
        if pilot_norm != 1:
            rcv_waveform[822:960] *= 1.7
            # (!!! 修正 !!!)
            rcv_waveform = rcv_waveform.reshape(-1, 960)
            rcv_waveform[:,:138] *= (pilot_norm / np.sqrt(1))
            rcv_waveform = rcv_waveform.flatten()
            
        rcv_symbols = mapper.signalToSymbols(rcv_waveform)
        channels_for_plot = mapper.get_mimo_channel(rcv_symbols) 

        # 3. SISO 均衡
        rcv_symbols_eq = mapper.equalize(rcv_symbols)

        # 4. 提取 Payload (SISO)
        rcv_payload = mapper.extractPayloads(rcv_symbols_eq)
        rcv_payload = rcv_payload[:PAYLOAD_LEN].astype(np.complex64)
        
        rcv_payload *= payload_norm # 反向正規化
        
        # --- C. (修改) 本地計算 ESNR ---
        print("Calculating ESNR...")
        sym_pow = np.mean(np.abs(gt_payload)**2)
        err_pow = np.mean(np.abs(gt_payload - rcv_payload)**2)
        esnr = 10*np.log10(sym_pow / err_pow)
        
        print(f"--- ESNR: {esnr:.2f} dB ---")

        # --- D. 更新繪圖 (SISO) ---
        
        # Channel plot
        ax1.clear()
        channel_mag = np.abs(np.mean(channels_for_plot, axis=-1).flatten())
        ax1.plot(channel_mag)
        ax1.set_xlim([0, 72])
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel('Subcarrier Index')
        ax1.set_ylabel('Channel Magnitude')
        ax1.set_title('SISO Channel Magnitude')

        # PSD
        ax2.clear()
        ax2.psd(rcv_waveform_full, NFFT=1024, Fs=Fs, scale_by_freq=False, linewidth=0.5, label='Chan 0')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectrum (dB)')
        ax2.set_title('Received PSD')

        # Constellation
        ax3.clear()
        ax3.scatter(np.real(rcv_payload), np.imag(rcv_payload), s=0.2, label='Received (Rx)')
        ax3.scatter(np.real(gt_payload), np.imag(gt_payload), s=2, color='orange', label='Ground Truth (GT)')
        ax3.set_xlim([-1.5, 1.5]) 
        ax3.set_ylim([-1.5, 1.5])
        ax3.set_xlabel('In-Phase')
        ax3.set_ylabel('Quadrature-Phase')
        ax3.set_title(f'QPSK Constellation | ESNR: {esnr:.2f} dB')
        ax3.legend()
        ax3.grid(True)
        ax3.set_aspect('equal')

        # 刷新 GUI
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01) 
        
except KeyboardInterrupt:
    print("\nStopping reception...")

finally:
    # 關閉串流
    stream_cmd_stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_streamer.issue_stream_cmd(stream_cmd_stop)
    
    plt.ioff()
    plt.close()
    print("Rx shut down.")
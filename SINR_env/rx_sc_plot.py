import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import config_ofdm as config

def plot_sc_metric(serial=""):
    # --- 1. 初始化 USRP ---
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
    
    print(f"--- S&C 視覺化示波器 ---")
    print(f"RX Gain: {config.RX_GAIN}")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    
    # 抓取足夠長的數據以包含完整的 Preamble + Data
    num_samps = 2000
    buff = np.zeros((1, num_samps), dtype=np.complex64)
    
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.RXMetadata()
    
    # --- 2. 準備繪圖 ---
    plt.ion()
    fig, (ax_raw, ax_metric) = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4)
    
    # 上圖：原始訊號振幅
    x_axis = np.arange(num_samps)
    line_raw, = ax_raw.plot(x_axis, np.zeros(num_samps), 'b-', lw=1)
    ax_raw.set_title("Raw Signal Amplitude")
    ax_raw.set_ylim(-0.1, 1.2)
    ax_raw.set_ylabel("|x[n]|")
    ax_raw.grid(True, alpha=0.5)
    
    # 下圖：Schmidl & Cox Metric (M)
    # 這裡 x 軸長度會少 L (因為做了延遲相關)
    L = len(config.SC_PREAMBLE) // 2
    metric_len = num_samps - L + 1
    line_metric, = ax_metric.plot(np.arange(metric_len), np.zeros(metric_len), 'r-', lw=1.5)
    
    ax_metric.set_title("Schmidl & Cox Timing Metric (M)")
    ax_metric.set_ylim(0, 1.1) # 理論最大值是 1.0
    ax_metric.set_ylabel("Metric Value")
    ax_metric.set_xlabel("Sample Index")
    
    # 畫一條 0.4 的門檻線
    ax_metric.axhline(y=0.4, color='g', linestyle='--', alpha=0.7, label='Threshold (0.4)')
    ax_metric.legend()
    ax_metric.grid(True, alpha=0.5)
    
    print(">> 開始繪圖... (顯示 '平台' 效應)")
    
    try:
        while True:
            # --- 3. 抓取快照 (Burst) ---
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            samps_recvd = 0
            while samps_recvd < num_samps:
                num = streamer.recv(buff[:, samps_recvd:], md)
                if md.error_code != uhd.types.RXMetadataErrorCode.none: break
                samps_recvd += num
            
            data = buff[0]
            
            # --- 4. 計算 S&C Metric ---
            # P(d) = sum(r[d] * r*[d+L])
            # R(d) = sum(|r[d+L]|^2)
            # M(d) = |P(d)|^2 / R(d)^2
            
            # 快速向量運算
            # delay_prod = r[d] * r*[d+L]
            delay_prod = data[:-L] * np.conj(data[L:])
            
            # 滑動加總 (Moving Sum)
            window = np.ones(L)
            P = np.convolve(delay_prod, window, mode='valid')
            
            # 能量計算
            energy = np.abs(data)**2
            R = np.convolve(energy, window, mode='valid')
            # R 需要跟 P 對齊 (取前段)
            R = R[:len(P)]
            
            # 計算 M (避免除以 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                M = (np.abs(P)**2) / (R**2 + 1e-10)
                M = np.nan_to_num(M) # 把 NaN 變 0
            
            # --- 5. 更新圖表 ---
            # 上圖
            amp = np.abs(data)
            line_raw.set_ydata(amp)
            
            # 檢查 Clipping
            if np.max(amp) > 0.98:
                ax_raw.set_title("Raw Signal (WARNING: Clipping!)", color='red')
            else:
                ax_raw.set_title("Raw Signal Amplitude", color='black')
                
            # 下圖
            # 補零以對齊長度 (如果 convolve 後變短)
            plot_M = np.zeros(metric_len)
            plot_M[:len(M)] = M
            line_metric.set_ydata(plot_M)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # --- 6. 暫停 (避免 Overflow) ---
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n停止繪圖")
    finally:
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="")
    args = parser.parse_args()
    plot_sc_metric(args.serial)
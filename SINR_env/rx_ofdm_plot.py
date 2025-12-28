import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import config_ofdm as config

def plot_ofdm_waveform(serial=""):
    # --- 1. 初始化 USRP ---
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
    
    print(f"--- 初始化繪圖 (OFDM 版) ---")
    print(f"裝置: {device_args}")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    usrp.set_rx_antenna("TX/RX")
    
    # 設定快照長度
    # 我們抓 2000 點，大約包含 20~25 個 OFDM 符號 (每個長度 80)
    # 這樣可以看清楚連續發送的波形結構
    num_samps = 2000
    recv_buffer = np.zeros((1, num_samps), dtype=np.complex64)
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_rx_stream(st_args)
    md = uhd.types.RXMetadata()
    
    # --- 2. 準備繪圖視窗 ---
    plt.ion() # 開啟互動模式
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4) # 調整子圖間距
    
    # 初始化線條 (預先建立物件，之後只更新數據，速度較快)
    x_axis = np.arange(num_samps)
    line_mag, = ax_time.plot(x_axis, np.zeros(num_samps), 'b-', lw=1)
    
    ax_time.set_title("Time Domain (Amplitude)", fontsize=12)
    ax_time.set_ylim(-0.1, 1.2) # 振幅通常在 0~1 之間
    ax_time.set_ylabel("Magnitude |x|")
    ax_time.set_xlabel("Sample Index")
    ax_time.grid(True, linestyle='--', alpha=0.6)
    
    # 頻域圖 X 軸 (頻率 MHz)
    freqs = np.fft.fftshift(np.fft.fftfreq(num_samps, 1/config.SAMPLE_RATE)) / 1e6
    line_fft, = ax_freq.plot(freqs, np.zeros(num_samps), 'm-', lw=1)
    
    ax_freq.set_title("Frequency Domain (Spectrum)", fontsize=12)
    ax_freq.set_ylim(-80, 5) # dB 範圍
    ax_freq.set_ylabel("Power (dB)")
    ax_freq.set_xlabel("Frequency (MHz)")
    ax_freq.grid(True, linestyle='--', alpha=0.6)
    
    print(">> 開始繪圖... (不會出現 O)")
    print(">> 請確保 tx_ofdm.py 正在執行中")
    
    try:
        while True:
            # --- 3. 請求數據 (Burst Mode) ---
            # 使用修正後的 StreamCMD 寫法，避免報錯
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            # --- 4. 接收數據 ---
            samps_recvd = 0
            timeout_counter = 0
            
            while samps_recvd < num_samps:
                num = streamer.recv(recv_buffer[:, samps_recvd:], md)
                
                # 簡單的超時保護
                if num == 0:
                    timeout_counter += 1
                    if timeout_counter > 1000: break
                
                if md.error_code != uhd.types.RXMetadataErrorCode.none:
                    # 忽略一些啟動時的錯誤
                    pass
                samps_recvd += num
            
            # 取出收到的數據
            data = recv_buffer[0, :]
            
            # --- 5. 繪圖更新 (這時候 USRP 已經停了，我們可以慢慢畫) ---
            
            # [時域更新]
            magnitude = np.abs(data)
            line_mag.set_ydata(magnitude)
            
            # 檢查是否過大 (Clipping Warning)
            if np.max(magnitude) > 1.0:
                ax_time.set_title("Time Domain (WARNING: Signal Saturated!)", color='red')
            else:
                ax_time.set_title("Time Domain (Amplitude)", color='black')

            # [頻域更新]
            # 加上 Blackman window 讓頻譜更好看
            window = np.blackman(num_samps)
            fft_data = np.fft.fftshift(np.fft.fft(data * window))
            psd = 20 * np.log10(np.abs(fft_data) + 1e-12)
            
            # 正規化顯示 (讓最高點大約在 0 dB)
            psd_norm = psd - np.max(psd)
            line_fft.set_ydata(psd_norm)
            
            # 刷新畫面
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # --- 6. 暫停 (關鍵步驟) ---
            # 這裡暫停 0.1 秒，讓 GUI 有時間響應，也讓 CPU 休息
            # 這就是避免 "O" 的秘訣
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
    plot_ofdm_waveform(args.serial)
import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import config

def live_plot_safe(serial=""):
    # --- 1. USRP 設定 ---
    device_args = "type=b200"
    if serial:
        device_args += f",serial={serial}"
    
    print(f"初始化繪圖接收機 (Safe Mode) | 裝置: {device_args}")
    usrp = uhd.usrp.MultiUSRP(device_args)
    
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    usrp.set_rx_antenna("TX/RX")
    
    num_samps_requested = 2000
    recv_buffer = np.zeros((1, num_samps_requested), dtype=np.complex64)
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_rx_stream(st_args)
    md = uhd.types.RXMetadata()
    
    # --- 2. 準備繪圖視窗 ---
    plt.ion() # 開啟互動模式
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 初始化線條物件
    x_axis = np.arange(num_samps_requested)
    line_mag, = ax_time.plot(x_axis, np.zeros(num_samps_requested), 'b-')
    
    ax_time.set_title("Time Domain (Snapshot Mode)")
    ax_time.set_ylim(-0.1, 1.1)
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)
    
    line_fft, = ax_freq.plot(x_axis, np.zeros(num_samps_requested), 'm-')
    ax_freq.set_title("Frequency Domain (PSD)")
    ax_freq.set_ylim(-100, 0)
    ax_freq.set_ylabel("Power (dB)")
    ax_freq.grid(True)
    
    print("開始繪圖... (不會出現 O)")
    print("按 Ctrl+C 停止")
    
    try:
        while True:
            # --- 關鍵修改：請求固定數量的數據 (Burst Request) ---
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps_requested
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            # 接收數據 (直到收滿 num_samps_requested)
            samps_recvd = 0
            while samps_recvd < num_samps_requested:
                num = streamer.recv(recv_buffer[:, samps_recvd:], md)
                
                if md.error_code != uhd.types.RXMetadataErrorCode.none:
                    print(f"Error: {md.error_code}")
                    break
                samps_recvd += num
            
            # --- 3. 數據處理與繪圖 (這時候 USRP 已經停了，慢慢畫沒關係) ---
            data = recv_buffer[0, :]
            
            # 更新時域
            magnitude = np.abs(data)
            line_mag.set_ydata(magnitude)
            
            # 自動調整 Y 軸 (如果訊號太強 Clipping)
            max_val = np.max(magnitude)
            if max_val > 1.0:
                ax_time.set_title("Time Domain (CLIPPING WARNING!)", color='red')
            else:
                ax_time.set_title("Time Domain", color='black')

            # 更新頻域
            fft_data = np.fft.fftshift(np.fft.fft(data))
            psd = 20 * np.log10(np.abs(fft_data) + 1e-12)
            psd -= np.max(psd) # 正規化
            line_fft.set_ydata(psd)
            
            # 畫出來並暫停一小段時間讓 GUI 更新
            plt.draw()
            plt.pause(0.05) # 這裡會暫停 0.05 秒，這就是你的「刷新率」
            
    except KeyboardInterrupt:
        print("\n停止繪圖")
    finally:
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="", help="B210 RX Serial")
    args = parser.parse_args()
    live_plot_safe(args.serial)
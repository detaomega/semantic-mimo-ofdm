import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import config_qpsk as config

def estimate_and_correct_cfo(rx_block, known_block):
    """
    簡單的相位修正：
    比較接收到的點和已知點的相位差，並消除線性趨勢 (CFO)
    """
    # 1. 計算每個點的相位差
    # rx = known * exp(j * theta)  ->  rx * conj(known) = |known|^2 * exp(j * theta)
    phase_diff = np.angle(rx_block * np.conj(known_block))
    
    # 2. 展開相位 (Unwrap) 以檢測線性趨勢
    phase_unwrapped = np.unwrap(phase_diff)
    
    # 3. 線性回歸 (Linear Regression) 找斜率
    # 斜率 = 頻率偏移 (CFO)
    # 截距 = 初始相位偏移
    x = np.arange(len(rx_block))
    slope, intercept = np.polyfit(x, phase_unwrapped, 1)
    
    # 4. 建立修正向量
    correction = np.exp(-1j * (slope * x + intercept))
    
    return rx_block * correction

def run_qpsk_vis(serial=""):
    device_args = "type=b200"
    if serial: device_args += f",serial={serial}"
    
    print(f"--- QPSK 接收機 (星座圖與 SINR) ---")
    print(f"RX Gain: {config.RX_GAIN} dB")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    
    num_samps = 4000 # 抓長一點，包含好幾個 Block
    buff = np.zeros((1, num_samps), dtype=np.complex64)
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.RXMetadata()
    
    # --- 繪圖設定 ---
    plt.ion()
    fig, (ax_const, ax_text) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 星座圖
    line_const, = ax_const.plot([], [], 'b.', markersize=2, alpha=0.6)
    line_ideal, = ax_const.plot(np.real(config.QPSK_SEQ[:100]), np.imag(config.QPSK_SEQ[:100]), 'r+', markersize=10, label='Ideal')
    ax_const.set_title("QPSK Constellation (De-rotated)")
    ax_const.set_xlim(-2, 2)
    ax_const.set_ylim(-2, 2)
    ax_const.grid(True)
    ax_const.legend()
    
    # 文字資訊區 (把 SINR 寫在圖上比較清楚)
    ax_text.axis('off')
    text_info = ax_text.text(0.1, 0.5, "Waiting...", fontsize=14)
    
    print(">> 開始接收... (請看彈出的視窗)")
    
    try:
        while True:
            # 1. 抓取數據
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps; stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            samps_recvd = 0
            while samps_recvd < num_samps:
                num = streamer.recv(buff[:, samps_recvd:], md)
                if md.error_code != uhd.types.RXMetadataErrorCode.none: break
                samps_recvd += num
            data = buff[0, :samps_recvd]
            
            # 2. 同步 (尋找 Sync Header)
            corr = np.correlate(data, config.SYNC_SEQ, mode='valid')
            peak_idx = np.argmax(np.abs(corr))
            peak_val = np.abs(corr[peak_idx])
            
            # 門檻 (如果太低代表全是雜訊)
            if peak_val > 10.0:
                # 3. 提取一個完整的 Block
                start = peak_idx
                end = start + config.BLOCK_LEN
                
                if end < len(data):
                    rx_block = data[start:end]
                    
                    # 4. 修正旋轉 (De-rotation) - 這是關鍵！
                    # 如果不修正，星座圖會變成甜甜圈
                    rx_corrected = estimate_and_correct_cfo(rx_block, config.QPSK_SEQ)
                    
                    # 5. 計算 EVM & SINR
                    # 誤差 = 修正後的點 - 理想的點
                    # 這裡要做一個振幅正規化 (把接收能量縮放到 1)
                    avg_power = np.mean(np.abs(rx_corrected))
                    rx_norm = rx_corrected / avg_power
                    
                    error = rx_norm - config.QPSK_SEQ # config 也是長度 1
                    noise_pwr = np.mean(np.abs(error)**2)
                    sig_pwr = 1.0 # 理想訊號功率為 1
                    
                    sinr = 10 * np.log10(sig_pwr / noise_pwr)
                    
                    # 6. 更新圖表
                    line_const.set_data(np.real(rx_norm), np.imag(rx_norm))
                    
                    status_color = "green" if sinr > 10 else "red"
                    info_str = f"Sync Strength: {peak_val:.1f}\nSINR: {sinr:.2f} dB"
                    text_info.set_text(info_str)
                    text_info.set_color(status_color)
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            else:
                text_info.set_text(f"No Sync (Peak={peak_val:.1f})\nAdjust Gain?")
                fig.canvas.draw()
                fig.canvas.flush_events()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stop")
    finally:
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="")
    args = parser.parse_args()
    run_qpsk_vis(args.serial)
import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import config_ofdm as config

def apply_cfo_correction(rx_packet):
    """
    CFO 校正 (與之前相同，為了讓正方形不要旋轉)
    """
    cp_part = rx_packet[0 : config.CP_LEN]
    tail_part = rx_packet[config.FFT_SIZE : config.TOTAL_LEN]
    phase_diff = np.sum(tail_part * np.conj(cp_part))
    angle = np.angle(phase_diff)
    cfo_est = angle / config.FFT_SIZE
    t = np.arange(len(rx_packet))
    correction_vector = np.exp(-1j * cfo_est * t)
    return rx_packet * correction_vector

def live_constellation_plot(serial=""):
    # --- 1. 初始化 USRP ---
    device_args = "type=b200"
    if serial: device_args += f",serial={serial}"
    
    print("--- OFDM 星座圖繪製 (Constellation Plot) ---")
    # 這裡預設 RX Gain 為 35，這是一個經驗上的甜蜜點
    RX_GAIN = 35 
    print(f"RX Gain: {RX_GAIN} dB")
    
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(RX_GAIN)
    usrp.set_rx_antenna("TX/RX")
    
    # 抓長一點，確保能抓到好幾個 Symbol
    num_samps = 10000 
    recv_buffer = np.zeros((1, num_samps), dtype=np.complex64)
    
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.RXMetadata()
    
    # --- 2. 準備繪圖視窗 ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 畫出理想的 QPSK 點 (紅色的 X)
    ideal_qpsk = config.qpsk_symbols
    ax.scatter(np.real(ideal_qpsk), np.imag(ideal_qpsk), c='red', marker='x', s=100, label='Ideal Pilot')
    
    # 畫出接收到的點 (藍色的點)
    # 我們先畫空的，之後用 set_offsets 更新
    scatter = ax.scatter([], [], c='blue', alpha=0.5, s=20, label='Received')
    
    # 設定座標軸範圍 (QPSK 點通常在 +/- 0.707)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title("OFDM Constellation (Waiting for Signal...)")
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--')
    
    print(">> 開始繪圖... (請確保 TX 正在發送)")
    
    try:
        while True:
            # --- 3. 接收數據 (快照) ---
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            
            samps_recvd = 0
            while samps_recvd < num_samps:
                num = streamer.recv(recv_buffer[:, samps_recvd:], md)
                if md.error_code != uhd.types.RXMetadataErrorCode.none: break
                samps_recvd += num
            
            data = recv_buffer[0, :samps_recvd]
            
            # --- 4. 訊號處理 ---
            # 同步
            corr = np.correlate(data, config.TIME_SYMBOL, mode='valid')
            peak_idx = np.argmax(np.abs(corr))
            peak_val = np.abs(corr[peak_idx])
            
            # 只有當訊號夠強時才更新畫面，不然畫面會一直閃爍亂跳
            if peak_val > 5.0:
                all_received_symbols = []
                current_idx = peak_idx
                
                # 嘗試連續解碼 5 個 OFDM 符號，累積點數讓正方形更明顯
                for _ in range(5):
                    if current_idx + config.TOTAL_LEN > len(data): break
                    
                    # 取出 Symbol
                    raw_symbol = data[current_idx : current_idx + config.TOTAL_LEN]
                    
                    # CFO 校正 (關鍵！不然正方形會轉圈圈)
                    corrected_symbol = apply_cfo_correction(raw_symbol)
                    
                    # 去 CP, FFT
                    symbol_no_cp = corrected_symbol[config.CP_LEN:]
                    freq_domain = np.fft.fft(symbol_no_cp) / np.sqrt(config.FFT_SIZE)
                    
                    # 取出有效子載波
                    indices = (config.OCCUPIED_BINS + config.FFT_SIZE) % config.FFT_SIZE
                    rx_pilots = freq_domain[indices]
                    
                    # 通道估測與等化 (Channel Estimation & Equalization)
                    # H = Rx / Tx_Known
                    # 簡單的做法：我們假設通道在這個瞬間是平坦的，或者直接除以 Pilot
                    # 這裡用 Zero-Forcing Equalizer
                    H_est = rx_pilots / config.qpsk_symbols 
                    
                    # 為了顯示穩定，我們可以用這 5 個符號的平均 H 來做等化
                    # 但為了簡單即時顯示，我們直接把接收到的點「除以通道」
                    # Rx_Equalized = Rx / H_est = Rx / (Rx/Tx) = Tx (理想狀態)
                    # 實際操作：我們需要一個參考點。
                    # 因為這範例整包都是 Pilot，所以我們可以直接拿 rx_pilots 跟 config.qpsk_symbols 比對
                    
                    # 這裡稍微偷吃步：我們畫的是「等化後」的結果
                    # 實際上我們應該用前一個符號估出的 H 來修正下一個
                    # 但因為我們是連續 Pilot，所以直接觀察 Rx 分佈即可
                    
                    # 為了把點「拉正」到紅色的 X 上，我們需要除以一個平均的相位/振幅旋轉
                    # 簡單估測：計算平均的 H
                    avg_H = np.mean(rx_pilots / config.qpsk_symbols)
                    rx_equalized = rx_pilots / avg_H
                    
                    all_received_symbols.extend(rx_equalized)
                    
                    current_idx += config.TOTAL_LEN
                
                # --- 5. 更新繪圖 ---
                if len(all_received_symbols) > 0:
                    points = np.array(all_received_symbols)
                    
                    # 更新藍色點的位置
                    scatter.set_offsets(np.c_[np.real(points), np.imag(points)])
                    
                    ax.set_title(f"Constellation (Peak: {peak_val:.1f}) - LOCKED")
                    
                    # 重新繪製
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            else:
                ax.set_title("Searching for Signal...")
                fig.canvas.draw()
                fig.canvas.flush_events()

            # --- 6. 休息 (防止 O) ---
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Stop")
    finally:
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="")
    args = parser.parse_args()
    live_constellation_plot(args.serial)
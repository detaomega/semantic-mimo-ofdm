import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import config_qpsk as config

def estimate_and_correct_cfo(rx_block, known_block):
    # (這部分負責把甜甜圈轉正)
    phase_diff = np.angle(rx_block * np.conj(known_block))
    phase_unwrapped = np.unwrap(phase_diff)
    x = np.arange(len(rx_block))
    slope, intercept = np.polyfit(x, phase_unwrapped, 1)
    correction = np.exp(-1j * (slope * x + intercept))
    return rx_block * correction

def demodulate_qpsk(complex_data):
    """
    QPSK 解調變 (Hard Decision)
    根據所在的象限判斷是 0, 1, 2, 3
    """
    demod_syms = np.zeros(len(complex_data), dtype=int)
    
    real = np.real(complex_data)
    imag = np.imag(complex_data)
    
    # Q1: (+, +) -> 0 (預設)
    
    # Q2: (-, +) -> 1
    idx_q2 = (real < 0) & (imag >= 0)
    demod_syms[idx_q2] = 1
    
    # Q3: (-, -) -> 2
    idx_q3 = (real < 0) & (imag < 0)
    demod_syms[idx_q3] = 2
    
    # Q4: (+, -) -> 3
    idx_q4 = (real >= 0) & (imag < 0)
    demod_syms[idx_q4] = 3
    
    return demod_syms

def run_qpsk_ber(serial=""):
    device_args = "type=b200"
    if serial: device_args += f",serial={serial}"
    
    print(f"--- QPSK BER 計算器 (修正版) ---")
    usrp = uhd.usrp.MultiUSRP(device_args)
    usrp.set_rx_rate(config.SAMPLE_RATE)
    usrp.set_rx_freq(uhd.types.TuneRequest(config.CENTER_FREQ))
    usrp.set_rx_gain(config.RX_GAIN)
    
    # 預先計算正確答案的 Symbol (0,1,2,3)
    tx_ground_truth = demodulate_qpsk(config.QPSK_SEQ)
    
    num_samps = 4000
    buff = np.zeros((1, num_samps), dtype=np.complex64)
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    md = uhd.types.RXMetadata()
    
    # 繪圖初始化
    plt.ion()
    fig, (ax_const, ax_text) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    line_const, = ax_const.plot([], [], 'b.', markersize=2, alpha=0.6)
    ax_const.set_title("QPSK Constellation")
    ax_const.set_xlim(-2, 2); ax_const.set_ylim(-2, 2)
    ax_const.grid(True)
    ax_text.axis('off')
    text_info = ax_text.text(0.1, 0.5, "Waiting...", fontsize=12)
    
    total_bits = 0
    total_errors = 0
    
    try:
        while True:
            # --- 修正部分開始 ---
            # 舊寫法: streamer.issue_stream_cmd(uhd.types.StreamCMD(..., ..., ...))
            # 新寫法: 分開設定屬性，確保相容性
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_samps
            stream_cmd.stream_now = True
            streamer.issue_stream_cmd(stream_cmd)
            # --- 修正部分結束 ---

            samps_recvd = 0
            while samps_recvd < num_samps:
                num = streamer.recv(buff[:, samps_recvd:], md)
                samps_recvd += num
            data = buff[0, :samps_recvd]
            
            # 同步
            corr = np.correlate(data, config.SYNC_SEQ, mode='valid')
            peak_idx = np.argmax(np.abs(corr))
            
            if np.abs(corr[peak_idx]) > 10.0:
                start = peak_idx
                end = start + config.BLOCK_LEN
                
                if end < len(data):
                    rx_block = data[start:end]
                    
                    # 1. 修正旋轉
                    rx_corrected = estimate_and_correct_cfo(rx_block, config.QPSK_SEQ)
                    
                    # 2. 正規化振幅
                    avg_power = np.mean(np.abs(rx_corrected))
                    rx_norm = rx_corrected / avg_power
                    
                    # 3. 計算 SINR
                    error_vec = rx_norm - config.QPSK_SEQ
                    sinr = 10 * np.log10(1 / np.mean(np.abs(error_vec)**2))
                    
                    # 4. 計算 BER (Symbol Error Rate)
                    rx_syms = demodulate_qpsk(rx_norm)
                    symbol_errors = np.sum(rx_syms != tx_ground_truth)
                    
                    total_errors += symbol_errors
                    total_bits += len(rx_syms)
                    
                    ber = 0.0
                    if total_bits > 0:
                        ber = total_errors / total_bits
                    
                    # 繪圖
                    line_const.set_data(np.real(rx_norm), np.imag(rx_norm))
                    
                    status_str = (
                        f"SINR: {sinr:.2f} dB\n"
                        f"Errors: {total_errors} / {total_bits} syms\n"
                        f"Symbol Error Rate: {ber:.6f}\n"
                        f"(理論上 >14dB 應該要是 0)"
                    )
                    text_info.set_text(status_str)
                    text_info.set_color("green" if ber < 1e-3 else "red")
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stop")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="")
    args = parser.parse_args()
    run_qpsk_ber(args.serial)
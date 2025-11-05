import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

# 假設 USRP_signal 和 generate... 函式存在於此或已被 import
from usrp_signal import USRP_signal
from utils import generateZadoffChuSymbols, generatePilotSymbols
    
class OFDM_FrameGenerator:
    
    # --- 內部常數 ---
    _PILOT_NORM = 4.0
    _SYNC_SCALE = 1.7 # 從 "神秘數字" 1.7 轉換而來
    # ----------------
    
    def __init__(self, num_subcarriers, DC_guard, symbols_per_slot, slots_per_frame, pilot_place, sync_place,
                 subcarrier_spacing, FFT_size, num_cp_samples, num_ex_cp_samples, sequential_mapping, initial_pad,
                 num_antenna, antenna_idx):
        
        self.num_subcarriers = num_subcarriers
        self.num_guards = FFT_size - num_subcarriers - 1 - DC_guard
        self.symbols_per_slot = symbols_per_slot
        self.slots_per_frame = slots_per_frame
        self.pilot_place = pilot_place
        self.sync_place = sync_place
        self.subcarrier_spacing = subcarrier_spacing
        self.FFT_size = FFT_size
        self.sampling_rate = subcarrier_spacing * FFT_size
        self.num_cp_samples = num_cp_samples # 9
        self.num_ex_cp_samples = num_ex_cp_samples # 10
        self.add_cp = num_ex_cp_samples - num_cp_samples # 1
        
        # 訊框結構: 1 個長 CP (138) + 6 個短 CP (137) = 138 + 822 = 960 samples/slot
        self.n_samples_per_slot = (self.FFT_size + self.num_ex_cp_samples) + \
                                  (self.symbols_per_slot - 1) * (self.FFT_size + self.num_cp_samples)
        self.n_samples_per_frame = self.n_samples_per_slot * self.slots_per_frame # 960 * 50 = 48000
        
        self.num_antenna = num_antenna
        self.antenna_idx = antenna_idx
        

        # Data map excl. guard bands
        self.payload_map = np.zeros((self.num_subcarriers, symbols_per_slot*slots_per_frame))

        # Attach pilot only to selected subcarrier wrt antenna ports
        self.payload_map[:,pilot_place::symbols_per_slot] = 3 # unused
        self.payload_map[antenna_idx::num_antenna,pilot_place::symbols_per_slot] = 1
        self.payload_map[:,(sync_place+1)*symbols_per_slot-1] = 2
        if initial_pad:
            self.payload_map[:,1:symbols_per_slot-1] = 3

        # Actual data map including guard bands
        self.data_map = np.ones([self.FFT_size, symbols_per_slot*slots_per_frame]) * 3

        self.dc_start_idx = FFT_size//2 - DC_guard//2
        self.dc_end_idx = FFT_size//2 + DC_guard//2
        self.leading_guard_end_idx = self.num_guards//2
        self.trailing_guard_start_idx = FFT_size-self.num_guards//2
        # print(self.dc_start_idx, self.dc_end_idx, self.leading_guard_end_idx, self.trailing_guard_start_idx)
        self.data_map[self.leading_guard_end_idx+1:self.dc_start_idx] = self.payload_map[:self.dc_start_idx-self.leading_guard_end_idx-1]
        self.data_map[self.dc_end_idx+1:self.trailing_guard_start_idx] = self.payload_map[self.trailing_guard_start_idx-self.dc_end_idx-1:]
        
        self.sync_idx = np.where(self.data_map == 2)
        self.pilot_idx = np.where(self.data_map == 1)
        self.data_idx = np.where(self.data_map == 0)
        
        self.reverse_idx = np.where(self.data_map.T == 0)

        self.sync_symbols = generateZadoffChuSymbols(self.sync_idx[0].size)
        sync_frame = np.zeros_like(self.data_map, dtype=np.complex64)
        sync_frame[self.sync_idx] = self.sync_symbols
        
        # --- (新增) 計算 Sync Symbol 的樣本偏移量 ---
        # (sync_place+1)*symbols_per_slot-1 = (0+1)*7-1 = 6 (第 7 個 symbol, 索引 6)
        sync_symbol_frame_idx = (self.sync_place+1)*self.symbols_per_slot-1 
        
        # 動態計算 Symbol 0 到 Symbol 6 (不含) 的總樣本數
        offset = 0
        for i in range(sync_symbol_frame_idx):
            symbol_in_slot = i % self.symbols_per_slot
            if symbol_in_slot == self.pilot_place: # 假設 pilot_place = 0
                offset += (self.FFT_size + self.num_ex_cp_samples) # 138 (Symbol 0)
            else:
                offset += (self.FFT_size + self.num_cp_samples) # 137 (Symbol 1-5)
        
        # 儲存這個偏移量 (138 + 5 * 137 = 823)
        self.sync_symbol_start_offset = offset
        # --- (新增結束) ---

        
        # --- (產生用於高效同步的「短同步訊號」 (137 samples)) ---
        
        # 1. 取得單一 Sync Symbol (頻域) (索引 6)
        sync_symbol_freq_domain = sync_frame[:, sync_symbol_frame_idx]

        # 2. IFFT
        sync_symbol_time = np.fft.fftshift(sync_symbol_freq_domain)
        sync_symbol_time = np.fft.ifft(sync_symbol_time, n=self.FFT_size, norm='ortho') * np.sqrt(self.FFT_size/self.num_subcarriers)
        
        # 3. 加 CP (Sync symbol 是第 7 個, 索引 6, 用短 CP)
        cp_len = self.num_cp_samples # 9
        cp = sync_symbol_time[-cp_len:]
        sync_symbol_with_cp = np.hstack([cp, sync_symbol_time])
        
        # 4. 應用 Tx 功率調整 (除法)
        sync_symbol_with_cp /= self._SYNC_SCALE
        
        # 5. 儲存
        self.short_sync_waveform = sync_symbol_with_cp.astype(np.complex64)
        # --- (修改結束) ---
        
        self.pilot = generatePilotSymbols(self.pilot_idx[0].size, QAM_order=4, seed=1030)
        self.ref_pilot_channel = np.ones_like(self.data_map, dtype=np.complex64)
        self.ref_pilot_channel[self.pilot_idx] = self.pilot
        
        self.control_iqs = np.zeros_like(self.data_map, dtype=np.complex64)
        self.control_iqs[self.sync_idx] = self.sync_symbols
        self.control_iqs[self.pilot_idx] = self.pilot
        self.num_data = np.count_nonzero(self.data_map==0)
        self.sequential_mapping = sequential_mapping
    def showMap(self):
        plt.figure(figsize=(14, 8))
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cm.viridis.N)
        img = plt.imshow(self.data_map, norm=norm, interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5, 3.5], ax=plt.gca(), shrink=0.8)
        cbar.set_ticklabels(["Data", "Pilot Symbols", "Synchronization Signal", "Guard band (DC)"])
        plt.xlabel('Time Domain Symbols')
        plt.ylabel('Subcarriers')
        plt.title('OFDM frame symbol map')
        plt.show()
        
    def mapToFrame(self, data):
        if self.num_data < data.size:
            print(f"data size is too big for the frame!: {self.num_data}")
            return
        elif self.num_data > data.size:
            new_data = np.zeros(self.num_data, dtype=np.complex64)
            new_data[:data.size] = data
            data = new_data
        data = data.flatten()
        
        if self.sequential_mapping:
            symbols = self.control_iqs.copy().T
            symbols[self.reverse_idx] = data
            symbols = symbols.T
        else:
            symbols = self.control_iqs.copy()
            symbols[self.data_idx] = data
            
        return symbols
        
    def symbolsToSignal(self, symbols):
        # IFFT
        signal_wo_cp = np.fft.fftshift(symbols, axes=0)
        signal_wo_cp = np.fft.ifft(signal_wo_cp, n=self.FFT_size, axis=0, norm='ortho') * np.sqrt(self.FFT_size/self.num_subcarriers)
        signal_wo_cp = np.transpose(signal_wo_cp) # Shape (num_total_symbols, FFT_size) = (350, 128)
        
        # --- (重構) ---
        # 根據 138 + 6*137 結構，逐一 Symbol 加入 CP 和功率調整
        n_symbols_total = self.symbols_per_slot * self.slots_per_frame
        signals_with_cp = []
        
        symbol_index_sync = (self.sync_place+1)*self.symbols_per_slot-1 # Sync symbol 的絕對索引 (6)

        for i in range(n_symbols_total):
            symbol_data = signal_wo_cp[i, :]
            symbol_in_slot = i % self.symbols_per_slot

            # 1. 決定 CP 長度
            if symbol_in_slot == self.pilot_place: # 假設 pilot_place 總是第一個 symbol (0)
                cp_len = self.num_ex_cp_samples # 10
            else:
                cp_len = self.num_cp_samples # 9
            
            cp = symbol_data[-cp_len:]
            symbol_with_cp = np.hstack([cp, symbol_data])

            # 2. 功率調整 (Tx)
            if symbol_in_slot == self.pilot_place:
                symbol_with_cp /= (self._PILOT_NORM / np.sqrt(self.num_antenna))
            
            if i == symbol_index_sync: # 只調整第一個 slot 的 sync symbol
                symbol_with_cp /= self._SYNC_SCALE

            signals_with_cp.append(symbol_with_cp)
        
        signal = np.concatenate(signals_with_cp).flatten().astype(np.complex64)
        # --- (重構結束) ---
        
        return USRP_signal(initial_symbols=signal,
                                tone_Fs=self.sampling_rate,
                                oversample_rate=1)
        
    def signalToSymbols(self, signal):
        if isinstance(signal, USRP_signal):
            signal = signal.signal
        
        # --- (重構) ---
        # 根據 138 + 6*137 結構，逐一 Symbol 移除 CP 和反向功率調整
        symbols_wo_cp = []
        current_idx = 0
        n_symbols_total = self.symbols_per_slot * self.slots_per_frame
        
        symbol_index_sync = (self.sync_place+1)*self.symbols_per_slot-1 # Sync symbol 的絕對索引 (6)
        
        for i in range(n_symbols_total):
            symbol_in_slot = i % self.symbols_per_slot

            # 1. 決定 Symbol 長度
            if symbol_in_slot == self.pilot_place:
                cp_len = self.num_ex_cp_samples # 10
                symbol_len = self.FFT_size + cp_len # 138
            else:
                cp_len = self.num_cp_samples # 9
                symbol_len = self.FFT_size + cp_len # 137
            
            symbol_with_cp = signal[current_idx : current_idx + symbol_len]

            # 2. 功率調整 (Rx - 反向)
            if symbol_in_slot == self.pilot_place:
                symbol_with_cp *= (self._PILOT_NORM / np.sqrt(self.num_antenna))

            if i == symbol_index_sync:
                symbol_with_cp *= self._SYNC_SCALE
            
            # 3. 移除 CP
            symbol_data = symbol_with_cp[cp_len:]
            symbols_wo_cp.append(symbol_data)
            
            current_idx += symbol_len
        
        signal_stacked = np.stack(symbols_wo_cp) # Shape (350, 128)
        # --- (重構結束) ---

        # FFT
        signal_transposed = np.transpose(signal_stacked) # Shape (128, 350)
        symbols = np.fft.fft(signal_transposed, n=self.FFT_size, axis=0, norm='ortho') / np.sqrt(self.FFT_size/self.num_subcarriers)
        symbols = np.fft.fftshift(symbols, axes=0)
        return symbols
    
    def extractPayloads(self, symbols):
        if self.sequential_mapping:
            symbols = symbols.T
            data = symbols[self.reverse_idx]
            symbols = symbols.T
        else:
            data = symbols[self.data_idx]
        return data
        
    def synchronize(self, rcv_signal):
        if isinstance(rcv_signal, USRP_signal):
            rcv_signal = rcv_signal.signal
            
        # --- (修改) ---
        try:
            # 舊的、對CFO敏感的方法:
            # corr = np.abs(scipy.signal.correlate(rcv_signal, self.short_sync_waveform, 'valid'))
            
            # 新的、對CFO穩健的方法：在「功率」上進行相關
            rcv_power = np.abs(rcv_signal)**2
            template_power = np.abs(self.short_sync_waveform)**2
            
            corr = scipy.signal.correlate(rcv_power, template_power, 'valid')
            
            # 找到最大峰值
            peak_idx = np.argmax(corr)
            
            # (關鍵修正) 
            # peak_idx 是「同步訊號」的開頭
            # 我們要回傳的是「訊框」的開頭
            est_idx = peak_idx - self.sync_symbol_start_offset
            
            # 確保索引是有效的
            if est_idx < 0:
                est_idx = -1

        except ValueError:
            est_idx = -1

        return est_idx
    def get_mimo_channel(self, symbols):
        '''
        return: estimated mimo channel with shape [num_subcarriers, num_tx, num_pilot_slots]
        '''
        # (此函式保持不變, 供 SISO 繪圖或未來 MIMO 擴展使用)
        mimo_ref_pilot = np.repeat(np.reshape(self.pilot, (self.num_subcarriers//self.num_antenna, -1)), self.num_antenna, axis=0)
        pilot_rcv = np.concatenate([
            symbols[self.leading_guard_end_idx+1:self.dc_start_idx,self.pilot_place::self.symbols_per_slot],
            symbols[self.dc_end_idx+1:self.trailing_guard_start_idx,self.pilot_place::self.symbols_per_slot]
        ], axis=0)

        mimo_est_channel = pilot_rcv / mimo_ref_pilot
        mimo_est_channel = np.reshape(mimo_est_channel, (self.num_subcarriers//self.num_antenna, self.num_antenna, -1))
        intp_mimo_est_channel = np.repeat(mimo_est_channel, self.num_antenna, axis=0)

        return intp_mimo_est_channel

    def mimo_zf_equalize(self, symbols, mimo_channel):
        # (此函式保持不變, 供未來 MIMO 擴展使用)
        '''
        symbols: (FFT_size, num_rx, num_slots)
        mimo_channel: (num_subcarriers, num_rx, num_tx, num_pilot_slots)
        '''
        _, num_rx, num_tx, _ = mimo_channel.shape
        est_channel = np.ones((self.FFT_size, num_rx, num_tx, self.symbols_per_slot*self.slots_per_frame), np.complex64)

        est_channel[self.leading_guard_end_idx+1:self.dc_start_idx, ..., self.pilot_place::self.symbols_per_slot] = \
            mimo_channel[:self.dc_start_idx-self.leading_guard_end_idx-1]
        est_channel[self.dc_end_idx+1:self.trailing_guard_start_idx, ..., self.pilot_place::self.symbols_per_slot] = \
            mimo_channel[self.trailing_guard_start_idx-self.dc_end_idx-1:] 

        # Linear interpolation
        est_channel = np.concatenate([est_channel, est_channel[...,-self.symbols_per_slot:]], axis=-1)

        i, j = self.data_idx
        d = np.reshape((j % self.symbols_per_slot) / self.symbols_per_slot, (-1, 1, 1))
        ref_idx = (j // self.symbols_per_slot) * self.symbols_per_slot
        data_channel = d * est_channel[i, ..., ref_idx] + (1-d) * est_channel[i, ..., ref_idx+self.symbols_per_slot]
        
        mimo_channel_intp = np.random.normal(0., 1., est_channel[..., :-self.symbols_per_slot].shape).astype(np.complex64)
        mimo_channel_intp[i, ..., j] = data_channel
        mimo_channel_intp = np.concatenate(
            (mimo_channel_intp[self.leading_guard_end_idx+1:self.dc_start_idx],
             mimo_channel_intp[self.dc_end_idx+1:self.trailing_guard_start_idx]), axis=0)
        mimo_channel_intp = np.transpose(mimo_channel_intp, (0, 3, 1, 2))
        
        mimo_channel_inv = np.linalg.inv(mimo_channel_intp)
        
        symbols_valid = np.concatenate(
            (symbols[self.leading_guard_end_idx+1:self.dc_start_idx],
             symbols[self.dc_end_idx+1:self.trailing_guard_start_idx]), axis=0)
        
        symbols_eq_valid = np.einsum('sitr,sri->tsi', mimo_channel_inv, symbols_valid)

        symbols_eq = np.zeros((num_rx, self.FFT_size, self.symbols_per_slot*self.slots_per_frame), np.complex64)
        symbols_eq[:, self.leading_guard_end_idx+1:self.dc_start_idx, :] = symbols_eq_valid[:, :self.dc_start_idx-self.leading_guard_end_idx-1, :]
        symbols_eq[:, self.dc_end_idx+1:self.trailing_guard_start_idx, :] = symbols_eq_valid[:, self.trailing_guard_start_idx-self.dc_end_idx-1:, :] 

        return symbols_eq

    def zf_precode(self, symbols, mimo_channel):
        # (此函式保持不變, 供未來 MIMO 擴展使用)
        '''
        symbols: (FFT_size, num_rx, num_slots)
        mimo_channel: (num_subcarriers, num_rx, num_tx, 1)
        '''
        sync_signal = symbols[..., (self.sync_place+1)*self.symbols_per_slot-1].T
        num_subcarriers, num_rx, num_tx, _ = mimo_channel.shape
        mimo_channel = np.broadcast_to(mimo_channel, (num_subcarriers, num_rx, num_tx, self.symbols_per_slot*self.slots_per_frame))
        
        est_channel = np.reshape(np.identity(num_rx), (1, num_rx, num_tx, 1)) + 0.0j
        est_channel = np.broadcast_to(est_channel, (self.FFT_size, num_rx, num_tx, self.symbols_per_slot*self.slots_per_frame)).copy()
        est_channel[self.leading_guard_end_idx+1:self.dc_start_idx] = mimo_channel[:self.dc_start_idx-self.leading_guard_end_idx-1]
        est_channel[self.dc_end_idx+1:self.trailing_guard_start_idx] = mimo_channel[self.trailing_guard_start_idx-self.dc_end_idx-1:] 
        est_channel = np.transpose(est_channel, (0, 3, 1, 2))
        
        mimo_channel_inv = np.linalg.inv(est_channel)
        norm_factor = np.sqrt(np.sum(mimo_channel_inv ** 2, axis=(-1, -2), keepdims=True))
        mimo_channel_inv /= norm_factor
        
        symbols_precoded = np.einsum('sitr,sri->tsi', mimo_channel_inv, symbols)
        symbols_precoded[..., (self.sync_place+1)*self.symbols_per_slot-1] = sync_signal

        return symbols_precoded

    def equalize(self, symbols):
        # (此函式保持不變, 用於 SISO 均衡)
        # TODO : Now hardcoded for pilot_place = 0
        est_channel = symbols / self.ref_pilot_channel
        
        # Nearest neighbor interpolation
        payload_map = np.ones((self.num_subcarriers, self.symbols_per_slot*self.slots_per_frame), np.complex64)
        payload_map[:self.dc_start_idx-self.leading_guard_end_idx-1] = est_channel[self.leading_guard_end_idx+1:self.dc_start_idx]
        payload_map[self.trailing_guard_start_idx-self.dc_end_idx-1:] = est_channel[self.dc_end_idx+1:self.trailing_guard_start_idx]
        
        valid_pilot = payload_map[self.antenna_idx::self.num_antenna,self.pilot_place::self.symbols_per_slot]
        intp_pilot = np.repeat(valid_pilot, self.num_antenna, axis=0)
        payload_map[:,self.pilot_place::self.symbols_per_slot] = intp_pilot
        
        est_channel[self.leading_guard_end_idx+1:self.dc_start_idx] = payload_map[:self.dc_start_idx-self.leading_guard_end_idx-1]
        est_channel[self.dc_end_idx+1:self.trailing_guard_start_idx] = payload_map[self.trailing_guard_start_idx-self.dc_end_idx-1:]

        # Interpolation (time idx)
        est_channel = np.hstack([est_channel, est_channel[:,-self.symbols_per_slot:]]) #TODO

        i, j = self.data_idx
        d = (j % self.symbols_per_slot) / self.symbols_per_slot
        ref_idx = (j // self.symbols_per_slot) * self.symbols_per_slot
        data_channel = d * est_channel[i, ref_idx] + (1-d) * est_channel[i, ref_idx+self.symbols_per_slot]
        
        interpolated_channel = np.ones_like(self.data_map, dtype=np.complex64)
        interpolated_channel[i, j] = data_channel
        symbols_eq = symbols / interpolated_channel
        
        return symbols_eq

    def get_channel(self, symbols):
        # (此函式保持不變)
        est_channel = symbols / self.ref_pilot_channel
        est_channel = np.hstack([est_channel, est_channel[:,-self.symbols_per_slot:]]) #TODO
        return np.mean(est_channel)
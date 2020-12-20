import numpy as np
import scipy.io.wavfile as wv
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import scipy.signal as sgn
import scipy.sparse.linalg as spla
import os
import glob

def stft(input_sound, fs, dft_size=512, hop_size=128, zero_pad=256, window='hann', plot=0):  
    if len(np.shape(input_sound)) == 1:
        # Forward transform
        input_sound_padded = np.zeros(len(input_sound) + dft_size - zero_pad + hop_size - len(input_sound)%hop_size)
        input_sound_padded[:len(input_sound)] = input_sound
        L = len(input_sound_padded) -  dft_size
        stft_output = np.zeros([int((dft_size)/2) + 1, int(L/hop_size)], dtype=complex)
        for i in range(0, int(L/hop_size)):
            seg = input_sound_padded[(i*hop_size):(i*hop_size + dft_size - zero_pad)]
            # Choose and apply window if selected
            if window == 'triangular':
                seg = np.multiply(seg, np.bartlett(dft_size - zero_pad))
            elif window == 'hann':
                seg = np.multiply(seg, np.hanning(dft_size - zero_pad))
            elif window == 'hamming':
                seg = np.multiply(seg, np.hamming(dft_size - zero_pad))
            elif window == 'kaiser':
                seg = np.multiply(seg, np.kaiser(dft_size - zero_pad, beta=0.8))
            # Change size variable so that it will pad zeros to input    
            s_stft = np.fft.rfft(seg, dft_size)            
            stft_output[:,i] = s_stft
        
            t_axis = np.arange(0, (int(L/hop_size)*hop_size)/fs, hop_size/fs)
            f_axis = np.arange(0, fs/2, fs/(2*(int((dft_size)/2) + 1)))

        if plot:
            # # Make zero elements equal to the smallest value to avoid log(0) for plotting
            min_nonzero = np.min(stft_output[np.nonzero(stft_output)])
            stft_output[stft_output == 0] = min_nonzero
    # 
            plt.pcolormesh(t_axis, f_axis, (np.abs(stft_output)**0.4), cmap='gist_gray_r')
            plt.xlabel('Time (sec)')
            plt.ylabel('Freq (Hz)')
            plt.title('DFT=%d HOP=%d PAD=%d WINDOW=%s' %(dft_size, hop_size, zero_pad, window))
            plt.grid(alpha=0.5)
            plt.colorbar()
            plt.show()

        return stft_output, t_axis, f_axis
    elif len(np.shape(input_sound)) == 2:
        # Inverse transform
        L = (np.shape(input_sound)[1]+1)*hop_size
        stft_output = np.zeros([int(L)+hop_size,])
        for i in range(0, int((L - dft_size)/hop_size)):
            s_stft = np.fft.irfft(input_sound[:,i], dft_size)
            if len(window) > 0:
                # Window should only cover nonzero part, not zero paddings
                if window == 'triangular':
                    w = np.bartlett(dft_size - zero_pad)
                elif window == 'hann':
                    w = np.hanning(dft_size - zero_pad)
                elif window == 'hamming':
                    w = np.hamming(dft_size - zero_pad)
                elif window == 'kaiser':
                    w = np.kaiser(dft_size - zero_pad, beta=0.8)
                # Pad window with zeros to make dimensions equal
                window_padded = np.concatenate([w, np.zeros([zero_pad,])])
                s_stft = np.multiply(s_stft, window_padded)
            # Overlap and add
            stft_output[i*hop_size:(i*hop_size + dft_size)] += s_stft
        
        return stft_output

class SourceSeparation:

    class Source:
        def __init__(self, file=None, speaker=None, train_ratio=0., test_ratio=0., fs=None):
            if train_ratio and test_ratio > 0:
                raise Exception('Data has to be either for training or test. Cannot be both.')
            self.speaker = speaker
            if fs is None:
                self.path = file
                fs, s = wv.read(self.path)
            else:
                s = file
                self.path = None
            # Force single channel
            if len(s.shape) > 1:
                print('%d audio channels. Choosing the first one, ignoring the rest.'%min(s.shape))
                if s.shape[0] > s.shape[1]:
                    s = s[:,0]
                else:
                    s = s[0,:]
            # Train or test parts
            if train_ratio > 0:
                s = s[:round(max(s.shape)*train_ratio)]                
            elif test_ratio > 0:
                s = s[-round(max(s.shape)*test_ratio):]

            s = s/max(abs(s))
            self.data = np.array(s, ndmin=1)
            self.fs = fs
            self.stft_out = None

        def resample(self, new_fs):
            self.data = librosa.resample(self.data, self.fs, new_fs)
            self.fs = new_fs
        def set_spectrogram(self, dft_size=512, hop_size=128, zero_pad=256, window='hann', plot=0):
            self.stft_out = stft(self.data, self.fs, dft_size, hop_size, zero_pad, window, plot)

    class Mixture:
        def __init__(self, sources):
            self.sources = sources
            self.fs = None
            self.data = None
            self.stft_out = None
            self._make_mixture()

        def _make_mixture(self):
            max_fs = 0
            for source in self.sources:
                max_fs = max(max_fs, source.fs)
            
            mixture = []
            for source in self.sources:
                if source.fs != max_fs:
                    source.resample(max_fs)
                if len(mixture)==0:
                    mixture = np.copy(source.data)
                else:
                    mixture = mixture[:min(len(source.data), len(mixture))]
                    mixture += source.data[:min(len(source.data), len(mixture))]
            
            self.fs = max_fs
            self.data = mixture/np.max(np.abs(mixture))

        def set_spectrogram(self, dft_size=512, hop_size=128, zero_pad=256, window='hann', plot=0):
            self.stft_out = stft(self.data, self.fs, dft_size, hop_size, zero_pad, window, plot)

    def __init__(self):
        self.sources = []
        self.mixture = None

    def add_sources_from_file(self, data_path, N_speaker, train_ratio=0., test_ratio=0.):
        speaker_IDs = os.listdir(data_path)
        if len(speaker_IDs) < N_speaker:
            raise Exception('Not enough speakers in the dataset. Reduce the number of speakers or add more data.')
        else:
            for i in range(N_speaker):
                spk = speaker_IDs[i]
                # add files from common reading
                # file_list = glob.glob(os.path.join(data_path, spk, spk+'c*.wav'))
                file_list = glob.glob(os.path.join(data_path, spk, '*.wav'))
                np.random.seed(i*2)
                file_idx = np.random.randint(0, len(file_list))
                self.sources.append(self.Source(file_list[file_idx], spk, train_ratio, test_ratio))

    def add_sources(self, src_array):
        for s in src_array:
            self.sources.append(s)

    def list_sources(self):
        for s in self.sources:
            print(s.speaker, s.path)

    def make_mixture(self, sources):
        mix = self.Mixture(sources)
        self.mixture = mix

    def make_interrupting_mixture(self, sources, t_segment=1):
        lengths = []
        max_fs = 0
        for s in sources:
            lengths.append(len(s.data))
            max_fs = max(max_fs, s.fs)
        last_idx=0

        mix = np.zeros([len(sources)*max(lengths)])
        speakers = []
        while min(lengths) > max_fs*t_segment:
            np.random.seed(last_idx)
            source_idx = np.random.randint(0, len(sources)+1)
            if source_idx < len(sources):
                seg_length = round(sources[source_idx].fs*t_segment)
                sig = sources[source_idx].data[(-lengths[source_idx]):(-lengths[source_idx]+seg_length)]
                lengths[source_idx] -= seg_length
            else:
                sig = np.zeros([round(max_fs*t_segment),])
                for ii, s in enumerate(sources):
                    seg_length = round(s.fs*t_segment)
                    sig += s.data[(-lengths[ii]):(-lengths[ii]+seg_length)]
                    lengths[ii] -= seg_length
            mix[last_idx:(last_idx + seg_length)] = sig
            last_idx += seg_length
            speakers.append(source_idx)
        return mix[:last_idx], speakers

    def separate_sources_unsupervised(self):
        N_sources = len(self.sources)
        stft_out = self.mixture.stft_out[0]
        X = np.abs(stft_out)
        W_nmf, H_nmf, cost = self.nmf(X, N_sources)
        separated_sources = []
        for i in range(N_sources):
            F_comp = np.outer(W_nmf[:,i], H_nmf[i,:])
            s_comp = stft(F_comp*np.exp(1j*np.angle(self.mixture.stft_out[0])), self.mixture.fs)
            separated_sources.append(s_comp)
        return separated_sources


    def nmf(self, X, N_source):
        N_f, N_t = X.shape
        np.random.seed(1531)
        W_nmf = np.random.uniform(size=[N_f, N_source])
        np.random.seed(6212)
        H_nmf = np.random.uniform(size=[N_source, N_t])

        X_nmf_hat = np.dot(W_nmf, H_nmf)
        X_nmf = X + np.finfo(np.double).eps
        N_nmf = X_nmf/X_nmf_hat # N_f x N_t
        cost = []
        for i in range(100):

            H_sum = np.array(np.sum(H_nmf, axis=1), ndmin=2) # (1 x N_source)
            # 👇 (N_f x N_source) = (N_f x N_source) * (N_f x N_t).(N_t x N_source) / (N_f x N_source)
            W_nmf = W_nmf * np.dot(N_nmf, H_nmf.T)/np.dot(np.ones([N_f, 1]), H_sum)

            X_nmf_hat = np.dot(W_nmf, H_nmf)
            N_nmf = X_nmf/X_nmf_hat # N_f x N_t
            W_sum = np.array(np.sum(W_nmf, axis=0), ndmin=2) # (1 x N_source)
            # 👇 (N_source x N_t) = (N_source x N_t) * (N_source x N_f).(N_f x N_t) / (N_f x N_source)
            H_nmf = H_nmf * np.dot(W_nmf.T, N_nmf) / np.dot(W_sum.T, np.ones([1, N_t]))

            X_nmf_hat = np.dot(W_nmf, H_nmf)
            N_nmf = X_nmf/X_nmf_hat # N_f x N_t

            cost.append(np.sum(X_nmf * np.log2(N_nmf) - X_nmf + X_nmf_hat))
            
        return W_nmf, H_nmf, cost
    def estimate_activations(self, X, W_nmf):
        # W_nmf: Learned training features, stacked horizontally. N_f x N_features*N_source matrix.
        N_f, N_t = X.shape
        np.random.seed(133)
        H_nmf = np.random.uniform(size=[W_nmf.shape[1], N_t])

        X_nmf_hat = np.dot(W_nmf, H_nmf)
        X_nmf = X + np.finfo(np.double).eps
        N_nmf = X_nmf/X_nmf_hat # N_f x N_t
        cost = []
        for i in range(100):
            H_sum = np.array(np.sum(H_nmf, axis=1), ndmin=2) # (1 x N_source)
            # 👇 (N_f x N_source) = (N_f x N_source) * (N_f x N_t).(N_t x N_source) / (N_f x N_source)
            # W_nmf = W_nmf * np.dot(N_nmf, H_nmf.T)/np.dot(np.ones([N_f, 1]), H_sum)

            X_nmf_hat = np.dot(W_nmf, H_nmf)
            N_nmf = X_nmf/X_nmf_hat # N_f x N_t
            W_sum = np.array(np.sum(W_nmf, axis=0), ndmin=2) # (1 x N_source)
            # 👇 (N_source x N_t) = (N_source x N_t) * (N_source x N_f).(N_f x N_t) / (N_f x N_source)
            H_nmf = H_nmf * np.dot(W_nmf.T, N_nmf) / np.dot(W_sum.T, np.ones([1, N_t]))

            X_nmf_hat = np.dot(W_nmf, H_nmf)
            N_nmf = X_nmf/X_nmf_hat # N_f x N_t

            cost.append(np.sum(X_nmf * np.log2(N_nmf) - X_nmf + X_nmf_hat))
            
        return H_nmf, cost


    def plt_features(self, W, N_features):
        # W: Feature matrix dictionary. W[speaker_name] is NFFT/2+1 x N_features matrix.
        # N_features:
        for s in self.sources:
            plt.figure(figsize=[10,10])
            for i in range(N_features):
                ax = plt.subplot2grid((round(N_features**0.5), round(N_features**0.5)+1), (i//(round(N_features**0.5)+1), i%(round(N_features**0.5)+1)))
                plt.plot(s.stft_out[2], W[s.speaker][:,i])
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.ylim([0,np.max(W[s.speaker])])
                if i == round(N_features**0.5)//2: plt.title(s.speaker)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

    def plot_nmf_with_spectrogram(self, N_sources=None):
        if N_sources == None:
            N_sources = len(self.sources)
        stft_out = self.mixture.stft_out[0]
        X = np.abs(stft_out)
        W_nmf, H_nmf, _ = self.nmf(X, N_sources)
        t_axis = self.mixture.stft_out[1]
        f_axis = self.mixture.stft_out[2]
        fig = plt.figure(figsize=[14,8])
        grid = plt.GridSpec(2*N_sources, 2*N_sources, wspace=0, hspace=0)
        main_ax = fig.add_subplot(grid[:-N_sources, N_sources:])
        plt.title('Spectrogram')
        plt.imshow(np.abs(self.mixture.stft_out[0])**0.3, aspect='auto',origin='lower', 
                   extent=(t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]), cmap='gist_gray_r')
        x_ax = fig.add_subplot(grid[N_sources+1, N_sources:2*N_sources])
        y_ax = fig.add_subplot(grid[:N_sources, 0])
        for i in range(N_sources):
            ax = plt.subplot(grid[:N_sources, i], sharey=main_ax)
            if i == 1: axl = ax
            plt.plot(-W_nmf[:,i], f_axis)
            plt.ylim([f_axis[0], f_axis[-1]]);plt.xlim([-7.5, 0.2])
            plt.ylabel('Frequency [Hz]');

            ax = plt.subplot(grid[i+N_sources, N_sources:2*N_sources], sharex=main_ax)
            if i == 1: axb = ax
            plt.plot(t_axis, H_nmf[i,:])
            plt.xlim([t_axis[0], t_axis[-1]]);plt.ylim([0, 10])
            plt.xlabel('Time');plt.ylabel(r'NMF Weights ($\mathbf{z}_i$)')
        for ax in fig.get_axes():
            ax.label_outer()
            
        axl.set_title(r'NMF Features ($\mathbf{w}_i$)')
        axb.set_xlabel('Time');axb.set_ylabel(r'NMF Weights ($\mathbf{z}_i$)')
        plt.show()



    

    # def pca(data, N_pca):
    #     # PCA input should be 0-mean
    #     d_mean = np.array(np.mean(data, 1), ndmin=2)
    #     d_zm = data - np.dot(d_mean.T, np.ones([1, data.shape[1]]))
    #     cov_d = np.dot(d_zm, d_zm.T)/(data.shape[1])
    #     eigvals, eigvecs = spla.eigsh(cov_d, k=N_pca, return_eigenvectors=True)
    #     W_pca = np.dot(np.diag(eigvals**-0.5), eigvecs.T)
    #     Z_pca = np.dot(W_pca, d_zm)
    #     return W_pca, Z_pca

 #    def ica(Z_pca):
    #     N_pca = Z_pca.shape[0]
    #     N_t = Z_pca.shape[0]
    #     W_ica = np.eye(N_pca) # np.random.normal(size=[N_pca, N_pca])
    #     delta_W_array = []
    #     N_batch = 20
    #     I = np.eye(N_pca)
    #     mu = 1e-5
    #     for _ in range(1500//(N_t//N_batch)):
    #         for i in range(N_t//N_batch):
    #             Z_pca_batch = Z_pca[:, (i*N_batch):((i+1)*N_batch)]
    #             delta_W = np.dot(N_batch*I - np.dot((2*np.tanh(Z_pca_batch)), Z_pca_batch.T), W_ica)
    #             W_ica = W_ica + mu*delta_W
    #             delta_W_array.append(np.mean(np.abs(delta_W)))

    #     Z_ica = np.dot(W_ica, Z_pca)
    #     return W_ica, Z_ica

    
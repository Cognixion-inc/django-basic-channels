from scipy.signal import butter, freqz, lfilter, square
from mne.filter import notch_filter
from brainflow.data_filter import DataFilter, WindowFunctions
from brainflow import BrainFlowInputParams, BoardIds, BoardShim
from time import sleep
from matplotlib import pyplot as plt
import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
import glob
import mne
import math
import scipy.stats
import os
import sklearn.cross_decomposition as sk

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")


def usbPort():
    usb_port = glob.glob("/dev/cu.usb*")
    if len(usb_port) != 1:
        print("Available Ports:")
        print(usb_port)
        print()
        print("Error in Detecting a Single USB Port")
        print("Please Enter Headset USB Port: ", end="")
        usb_port = input()
    else:
        usb_port = usb_port[0]

    print()
    print("Saved USB Port: " + usb_port)

    return usb_port

def startSession():
    params = BrainFlowInputParams()
    params.serial_port = usbPort()
    board_id = BoardIds.CNX_BOARD.value
    board = BoardShim(board_id, params)

    board.prepare_session()
    board.config_board("d")

    eeg_channels = board.get_eeg_channels(board_id)
    sampling_rate = board.get_sampling_rate(board_id)
    print("Connected to Board...")

    board.start_stream()
    sleep(7)
    data = board.get_board_data()
    print("Ready to Stream!")

    return board, eeg_channels, sampling_rate

def endSession(board):
    board.stop_stream()
    board.release_session()
    print("Board Session Ended!")

def preprocessing(data, eeg_chx, srate):

    min_sample = len(data[0][0])
    start = round(0.5 * srate)

    for i in range(1, len(data)):
        if min_sample > len(data[i][0]):
            min_sample = len(data[i][0])

    filt_data = np.zeros((len(data), len(eeg_chx), min_sample-start))
    trl = 0

    for trls in data:
        trial = np.zeros((len(eeg_chx), len(trls[0])))
        ch = 0
        for chx in eeg_chx:
            trial[ch] = trls[chx]
            ch += 1

        trial_notch = notch_filter(trial, srate, [60.0], method='iir')

        fs = srate
        lowcut = 5
        highcut = 35
        order = 4

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        trial_bandpass = lfilter(b, a, trial_notch)
        filt_data[trl] = trial_bandpass[:, start:min_sample]
        trl += 1

    return filt_data

def psd(data, eeg_chx, ch_names, srate, condition, graph=True, display=False):
    nfft = 1024
    psd = [[[] for i in range(len(eeg_chx))] for j in range(data.shape[0])]
    trl = 0
    for trial in data:
        for chx in range(len(trial)):
            psd[trl][chx] = np.array(DataFilter.get_psd_welch(trial[chx], nfft, nfft // 2, srate,
                                                        WindowFunctions.HAMMING.value))
        trl += 1
    if graph:
        high = round((nfft * 35)/srate)
        harmonics = np.zeros(shape=(4, 2))
        harmonics[0, 0] = 5
        harmonics[0, 1] = round((nfft * 5)/srate)
        harmonics[1, 0] = 10
        harmonics[1, 1] = round((nfft * 10)/srate)
        harmonics[2, 0] = 20
        harmonics[2, 1] = round((nfft * 20)/srate)
        harmonics[3, 0] = 30
        harmonics[3, 1] = round((nfft * 30)/srate)
        trl = 1
        for trial in psd:
            rows = math.ceil(len(eeg_chx)/2)
            fig, ax = plt.subplots(rows, 2, figsize=(10, 10))
            title = condition + \
                " Epoch Power Spectral Density (Trial " + str(trl) + ")"
            fig.suptitle(title)
            clrs = ['blue', 'red', 'orange', 'purple',
                'pink', 'green', 'brown', 'gray']
            lg_clrs = ['bx', 'gx', 'rx', 'kx']
            for r in range(rows):
                for c in range(2):
                    if len(eeg_chx) % 2 != 0 and r == rows - 1 and c == 1:
                        break
                    elif rows == 1:
                        ax[c].plot(trial[r*2+c][1, 0:high], trial[r*2+c]
                                   [0, 0:high], str('tab:' + clrs[r*2+c]))
                        space = 0.3 * \
                            (np.max(trial[r*2+c][0, 0:high]) -
                             np.min(trial[r*2+c][0, 0:high]))
                        ax[c].set_ylim(
                            (0, (np.max(trial[r*2+c][0, 0:high]) + 2*space)))
                        h = 0
                        for harmonic in harmonics:
                            if harmonic[0] == 8:
                                ax[c].plot(trial[r*2+c][1, int(harmonic[1])], trial[r*2+c][0, int(harmonic[1])]+space,
                                              lg_clrs[h], label="SSVEP")
                            if harmonic[0] == 12:
                                ax[c].plot(trial[r*2+c][1, int(harmonic[1])], trial[r*2+c][0, int(harmonic[1])]+space,
                                              lg_clrs[h], label="SSMVEP")
                            else:
                                ax[c].plot(trial[r*2+c][1, int(harmonic[1])], trial[r*2+c][0, int(harmonic[1])]+space,
                                              lg_clrs[h])
                            h += 1
                        ch_ind = r*2+c
                        ch_title = "Channel " + ch_names[ch_ind]
                        ax[c].set_title(ch_title)
                        ax[c].legend()
                    else:
                        ax[r, c].plot(trial[r*2+c][1, 0:high], trial[r*2+c]
                                      [0, 0:high], str('tab:' + clrs[r*2+c]))
                        space = 0.3 * \
                            (np.max(trial[r*2+c][0, 0:high]) -
                             np.min(trial[r*2+c][0, 0:high]))
                        ax[r, c].set_ylim(
                            (0, (np.max(trial[r*2+c][0, 0:high]) + 2*space)))
                        h = 0
                        for harmonic in harmonics:
                            ax[r, c].plot(trial[r*2+c][1, int(harmonic[1])], trial[r*2+c][0, int(harmonic[1])]+space,
                                          lg_clrs[h], label=str(int(harmonic[0])) + ' Hz')
                            h += 1
                        ch_ind = r*2+c
                        ch_title = "Channel " + ch_names[ch_ind]
                        ax[r, c].set_title(ch_title)
                        ax[r, c].legend()
            plt.tight_layout()
            fig_title = 'Results/' + condition + '_Trial_' + str(trl) + '.jpeg'
            fig.savefig(fig_title)
            if not display:
                plt.close()
            trl += 1

    return psd

class CCA(object):

    def __init__(self, frequencies, fs=250):
        '''
            Initializes CCA object
            frequencies: array of frequencies to be matched
            fs: sampling rate
        '''
        self.targets = len(frequencies)
        self.frequencies = frequencies
        self.fs = fs
    
    def refSignals(self, samples, freq):
        '''
            PURPOSE: Generates sinusoidal reference signals for given frequency
            
            samples: length of the EEG data
            freq: frequency of reference signals
        '''
        
        refSignals = np.zeros(shape=(samples,4))
        
        timePoints = np.arange(0, samples/self.fs, 1.0/self.fs)
        f0 = 2 * np.pi * freq * timePoints
        f1 = 2 * f0
        
        refSignals[:,0] = np.cos(f0)
        refSignals[:,1] = np.cos(f1)
        refSignals[:,2] = np.sin(f0)
        refSignals[:,3] = np.sin(f1)
        
        return refSignals
    
        
    def classify(self, data): 
        '''
            PURPOSE: Performs CCA to determine frequency of flashing target
            
            data: EEG data - assumes format to be num of channels x num of samples 
        '''
        numSamples = data.shape[0]
        numChannels = data.shape[1]
        
        cca = sk.CCA(n_components=1)
        ccaCoeff = np.zeros(shape=(self.targets,numChannels))
        pVal = np.zeros(shape=(self.targets,numChannels))
                            
        for target in range(self.targets):
            for ch in range(numChannels):
                refSigs = self.refSignals(numSamples, self.frequencies[target])
                cca.fit(data[:,ch:ch+1], refSigs)
                X_c, Y_c = cca.transform(data[:,ch:ch+1], refSigs)
                X_c = X_c.flatten()
                Y_c = Y_c.flatten()
                ccaCoeff[target][ch], pVal[target][ch] = scipy.stats.pearsonr(X_c, Y_c)
                
        if numChannels > 1:
            ccaCoeff = np.around(np.average(ccaCoeff, axis=1), 4)
            pVal = np.around(np.average(pVal, axis=1), 4)
            
        index = int(np.where(ccaCoeff == np.max(ccaCoeff))[0])
        
        return int(self.frequencies[index]), ccaCoeff, pVal

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.room_group_name = 'test'

        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name,
            self.channel_name
        )

        self.accept()

        self.tr = 0
        self.num_trials = 3
        self.target_freqs = [10, 12, 20, 24]
        self.ch_names = ['PO3', 'O1', 'Oz', 'O2', 'PO4', 'POz']
        self.ssvep_data = [[] for x in range(self.num_trials)]
        self.baseline_data = [[] for x in range(self.num_trials)]
        self.board, self.eeg_chx, self.srate = startSession()
        self.eeg_chx = self.eeg_chx[:-2]

        directory = './Results/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        if (message == 'Ready to Begin!'):
            async_to_sync(self.channel_layer.group_send)(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': message
                }
            )
        elif (message == 'Start!'):
            trial = "Trial " + str(self.tr+1)
            if self.tr == 0:
                _ = self.board.get_board_data()
            else:
                self.baseline_data[self.tr-1] = np.array(self.board.get_board_data())
                baseline_name = "Results/Baseline_RawData_Trial_" + str(self.tr) + ".csv"
                pd.DataFrame(self.baseline_data[self.tr-1]).to_csv(
                    baseline_name, sep='\t', index_label="Channel")

            async_to_sync(self.channel_layer.group_send)(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': trial
                }
            )
        elif (message == 'Done!'):
            self.ssvep_data[self.tr] = np.array(self.board.get_board_data())
            ssvep_name = "Results/SSVEP_RawData_Trial_" + str(self.tr+1) + ".csv"
            pd.DataFrame(self.ssvep_data[self.tr]).to_csv(
                ssvep_name, sep='\t', index_label="Channel")

            self.tr += 1
        elif (message == 'Complete!'):
            self.ssvep_data[self.tr] = np.array(self.board.get_board_data())
            ssvep_name = "Results/SSVEP_RawData_Trial_" + str(self.tr+1) + ".csv"
            pd.DataFrame(self.ssvep_data[self.tr]).to_csv(
                ssvep_name, sep='\t', index_label="Channel")

            sleep(5)

            self.baseline_data[self.tr] = np.array(self.board.get_board_data())
            baseline_name = "Results/Baseline_RawData_Trial_" + str(self.tr+1) + ".csv"
            pd.DataFrame(self.baseline_data[self.tr]).to_csv(
                baseline_name, sep='\t', index_label="Channel")

            endSession(self.board)

            # preprocessing
            ssvep_filter = preprocessing(
                self.ssvep_data, self.eeg_chx, self.srate)
            baseline_filter = preprocessing(
                self.baseline_data, self.eeg_chx, self.srate)

            # psd
            ssvep_psd = psd(ssvep_filter, self.eeg_chx,
                            self.ch_names, self.srate, "SSVEP")
            baseline_psd = psd(baseline_filter, self.eeg_chx,
                            self.ch_names, self.srate, "Baseline", graph=False)

            # cca
            ssvep_res = np.zeros(shape=(len(ssvep_filter), 1))
            ssvep_coeff = np.zeros(shape=(len(ssvep_filter), len(self.target_freqs)))
            ssvep_pval = np.zeros(shape=(len(ssvep_filter), len(self.target_freqs)))
            ssvep_cca_res = CCA(self.target_freqs, self.srate)
            trl = 0

            for trial in ssvep_filter:
                ssvep_res[trl], ssvep_coeff[trl], ssvep_pval[trl] = ssvep_cca_res.classify(
                    trial.T)
                trl += 1

            ind = [x for x in range(1, self.num_trials+1)]
            cols = [str(x) + ' Hz' for x in self.target_freqs]
            ssvep_cca = pd.DataFrame(ssvep_coeff, index=ind, columns=cols)
            ssvep_cca_pval = pd.DataFrame(ssvep_pval, index=ind, columns=cols)

            trl = 1
            for res in ssvep_res:
                ssvep_cca.at[trl, 'Result'] = str(int(res)) + " Hz"
                trl += 1

            ssvep_cca.to_csv("Results/SSVEP_CCA.txt", sep='\t', index_label="Trial")
            ssvep_cca_pval.to_csv("Results/SSVEP_PVAL.txt",
                                sep='\t', index_label="Trial")

            # erp activity 
            info = mne.create_info(self.ch_names, self.srate, ch_types=['eeg']*len(self.eeg_chx))
            info.set_montage('standard_1020')

            ssvep_ave = np.mean(ssvep_filter, axis=0)
            ssvep_evoked = mne.EvokedArray(ssvep_ave, info, comment='ssvep', nave=self.num_trials)

            fig_ssvep = ssvep_evoked.plot_topomap(ch_type='eeg', sensors='r+', size=2, show=False, 
                                            show_names=True, colorbar=True)
            fig_ssvep.savefig("Results/SSVEP_Average_ERP_Activity.jpeg")

            # snr
            foi = 10 # fundamental freq
            foi_peak = round((1024 * foi)/self.srate)
            snr_ssvep = np.zeros((self.num_trials, len(self.eeg_chx)))

            for trl in range(self.num_trials):
                for chx in range(len(self.eeg_chx)):
                    bsln_val = baseline_psd[trl][chx][0, foi_peak]
                    ssvep_val = ssvep_psd[trl][chx][0, foi_peak]
                    snr_ssvep[trl][chx] = round(ssvep_val/bsln_val, 3)
                    
            cols = [x for x in self.ch_names]
            snr_vep = pd.DataFrame(snr_ssvep, index=ind, columns=cols)
            snr_vep['Average'] = np.around(np.average(snr_ssvep, axis=1), 3)

            snr_vep.to_csv("Results/SSVEP_SNR.txt", sep='\t', index_label="Trial")

            async_to_sync(self.channel_layer.group_send)(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': 'Overt Assessment Complete!'
                }
            )

        print(message)

    def chat_message(self, event):
        message = event['message']

        self.send(text_data=json.dumps({
            'type':'chat',
            'message':message
        }))
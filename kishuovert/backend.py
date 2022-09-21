from asgiref.sync import async_to_sync
from brainflow.data_filter import DataFilter, WindowFunctions
from brainflow import BrainFlowInputParams, BoardIds, BoardShim
from channels.generic.websocket import WebsocketConsumer
from matplotlib import pyplot as plt
from mne.filter import notch_filter
from scipy.signal import butter, lfilter
from time import sleep, time

import glob
import json
import matplotlib
import mne
import math
import numpy as np
import os
import pandas as pd
import scipy.stats
import sklearn.cross_decomposition as sk

matplotlib.use("Qt5Agg")


class Board(object):

    def __init__(self):
        self.params = BrainFlowInputParams()
        self.board_id = BoardIds.CNX_BOARD.value
        self.params.serial_port = self.usbPort()
        self.board = BoardShim(self.board_id, self.params)

    def usbPort(self):
        usbPort = glob.glob("/dev/cu.usb*")
        if (len(usbPort) != 1):
            print("Available Ports:")
            print(usbPort)
            print()
            print("Error in Detecting a Single USB Port")
            print("Please Enter Headset USB Port: ", end="")
            usbPort = input()
        else:
            usbPort = usbPort[0]

        return usbPort

    def startSession(self):
        self.board.prepare_session()
        self.board.config_board("d")

        self.eegChx = self.board.get_eeg_channels(self.board_id)
        self.sRate = self.board.get_sampling_rate(self.board_id)
        print("Connected to Board...")

        self.board.start_stream()
        sleep(7)
        data = self.board.get_board_data()
        print("Ready to Stream!")

        return self.board, self.eegChx, self.sRate

    def endSession(self):
        self.board.stop_stream()
        self.board.release_session()
        print("Board Session Ended!")


class Processing(object):

    def __init__(self):
        self.numTrials = 3
        self.startTrial = []
        self.endTrial = []
        self.targetFreqs = [10, 12, 20, 24]
        self.chNames = ['PO3', 'O1', 'Oz', 'O2', 'PO4', 'POz']
        self.ssvepData = [[] for x in range(self.numTrials)]
        self.baselineData = [[] for x in range(self.numTrials)]

        directory = './Results/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        self.cnxBoard = Board()
        self.board, self.eegChx, self.sRate = self.cnxBoard.startSession()
        self.eegChx = self.eegChx[:-2]

    def begin(self):
        _ = self.board.get_board_data()
        self.startTime = round(time()*1000)

    def timing(self, trialTime, cond, lastTrial=False):
        tTime = ((trialTime - self.startTime)/1000) * self.sRate
        if cond == 'start':
            self.startTrial.append(round(tTime))
        elif cond == 'end':
            self.endTrial.append(round(tTime))

        if lastTrial:
            sleep(5)
            self.allData = np.array(self.board.get_board_data())
            self.startTrial.append(len(self.allData[0]))
            self.cnxBoard.endSession()
            self.createTrials()
            self.processingPipeline()

    def createTrials(self):
        for i in range(len(self.endTrial)):
            self.ssvepData[i] = self.allData[:, self.startTrial[i]:self.endTrial[i]]
            self.baselineData[i] = self.allData[:, self.endTrial[i]:self.startTrial[i+1]]
            ssvep_name = "Results/SSVEP_RawData_Trial_" + str(i+1) + ".csv"
            baseline_name = "Results/Baseline_RawData_Trial_" + str(i+1) + ".csv"
            pd.DataFrame(self.ssvepData[i]).to_csv(ssvep_name, sep='\t', index_label="Channel")
            pd.DataFrame(self.baselineData[i]).to_csv(baseline_name, sep='\t', index_label="Channel")

    def preprocessing(self, data):
        min_sample = len(data[0][0])
        start = round(0.5 * self.sRate)

        for i in range(1, len(data)):
            if min_sample > len(data[i][0]):
                min_sample = len(data[i][0])

        filt_data = np.zeros((len(data), len(self.eegChx), min_sample-start))
        trl = 0

        for trls in data:
            trial = np.zeros((len(self.eegChx), len(trls[0])))
            ch = 0
            for chx in self.eegChx:
                trial[ch] = trls[chx]
                ch += 1

            trial_notch = notch_filter(trial, self.sRate, [60.0], method='iir')

            fs = self.sRate
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

    def psd(self, data, condition, graph=True, display=False):
        nfft = 1024
        psd = [[[] for i in range(len(self.eegChx))] for j in range(data.shape[0])]
        trl = 0
        for trial in data:
            for chx in range(len(trial)):
                psd[trl][chx] = np.array(DataFilter.get_psd_welch(trial[chx], nfft, nfft // 2, self.sRate,
                                                                WindowFunctions.HAMMING.value))
            trl += 1
        if graph:
            high = round((nfft * 35)/self.sRate)
            harmonics = np.zeros(shape=(4, 2))
            harmonics[0, 0] = 5
            harmonics[0, 1] = round((nfft * 5)/self.sRate)
            harmonics[1, 0] = 10
            harmonics[1, 1] = round((nfft * 10)/self.sRate)
            harmonics[2, 0] = 20
            harmonics[2, 1] = round((nfft * 20)/self.sRate)
            harmonics[3, 0] = 30
            harmonics[3, 1] = round((nfft * 30)/self.sRate)
            trl = 1
            for trial in psd:
                rows = math.ceil(len(self.eegChx)/2)
                fig, ax = plt.subplots(rows, 2, figsize=(10, 10))
                title = condition + \
                    " Epoch Power Spectral Density (Trial " + str(trl) + ")"
                fig.suptitle(title)
                clrs = ['blue', 'red', 'orange', 'purple',
                        'pink', 'green', 'brown', 'gray']
                lg_clrs = ['bx', 'gx', 'rx', 'kx']
                for r in range(rows):
                    for c in range(2):
                        if len(self.eegChx) % 2 != 0 and r == rows - 1 and c == 1:
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
                            ch_title = "Channel " + self.chNames[ch_ind]
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
                            ch_title = "Channel " + self.chNames[ch_ind]
                            ax[r, c].set_title(ch_title)
                            ax[r, c].legend()
                plt.tight_layout()
                fig_title = 'Results/' + condition + '_Trial_' + str(trl) + '.jpeg'
                fig.savefig(fig_title)
                if not display:
                    plt.close()
                trl += 1

        return psd

    def ccaRefSignals(self, samples, freq):
        refSignals = np.zeros(shape=(samples, 4))
        timePoints = np.arange(0, samples/self.sRate, 1.0/self.sRate)
        f0 = 2 * np.pi * freq * timePoints
        f1 = 2 * f0

        refSignals[:, 0] = np.cos(f0)
        refSignals[:, 1] = np.cos(f1)
        refSignals[:, 2] = np.sin(f0)
        refSignals[:, 3] = np.sin(f1)

        return refSignals

    def ccaClassify(self, data):
        targets = len(self.targetFreqs)
        numSamples = data.shape[0]
        numChannels = data.shape[1]

        cca = sk.CCA(n_components=1)
        ccaCoeff = np.zeros(shape=(targets, numChannels))
        pVal = np.zeros(shape=(targets, numChannels))

        for target in range(targets):
            for ch in range(numChannels):
                refSigs = self.ccaRefSignals(
                    numSamples, self.targetFreqs[target])
                cca.fit(data[:, ch:ch+1], refSigs)
                X_c, Y_c = cca.transform(data[:, ch:ch+1], refSigs)
                X_c = X_c.flatten()
                Y_c = Y_c.flatten()
                ccaCoeff[target][ch], pVal[target][ch] = scipy.stats.pearsonr(
                    X_c, Y_c)

        if numChannels > 1:
            ccaCoeff = np.around(np.average(ccaCoeff, axis=1), 4)
            pVal = np.around(np.average(pVal, axis=1), 4)

        index = int(np.where(ccaCoeff == np.max(ccaCoeff))[0])

        return int(self.targetFreqs[index]), ccaCoeff, pVal

    def cca(self, data):
        ssvepResult = np.zeros(shape=(len(data), 1))
        ssvepCoeff = np.zeros(shape=(len(data), len(self.targetFreqs)))
        ssvepPVal = np.zeros(shape=(len(data), len(self.targetFreqs)))
        trl = 0

        for trial in data:
            ssvepResult[trl], ssvepCoeff[trl], ssvepPVal[trl] = self.ccaClassify(trial.T)
            trl += 1

        ind = [x for x in range(1, self.numTrials+1)]
        cols = [str(x) + ' Hz' for x in self.targetFreqs]
        ssvepCCA = pd.DataFrame(ssvepCoeff, index=ind, columns=cols)
        ssvepCCAPVal = pd.DataFrame(ssvepPVal, index=ind, columns=cols)

        trl = 1
        for res in ssvepResult:
            ssvepCCA.at[trl, 'Result'] = str(int(res)) + " Hz"
            trl += 1

        ssvepCCA.to_csv("Results/SSVEP_CCA.txt",sep='\t', index_label="Trial")
        ssvepCCAPVal.to_csv("Results/SSVEP_PVAL.txt", sep='\t', index_label="Trial")

    def erp(self, data):
        info = mne.create_info(self.chNames, self.sRate, ch_types=['eeg']*len(self.eegChx))
        info.set_montage('standard_1020')

        ssvepAve = np.mean(data, axis=0)
        ssvepEvoked = mne.EvokedArray(ssvepAve, info, comment='ssvep', nave=self.numTrials)

        figSSVEP = ssvepEvoked.plot_topomap(ch_type='eeg', sensors='r+', size=2, show=False, show_names=True, colorbar=True)
        figSSVEP.savefig("Results/SSVEP_Average_ERP_Activity.jpeg")

    def snr(self, ssvepData, baselineData):
        foi = 10  # fundamental freq
        foiPeak = round((1024 * foi)/self.sRate)
        snrSSVEP = np.zeros((self.numTrials, len(self.eegChx)))

        for trl in range(self.numTrials):
            for chx in range(len(self.eegChx)):
                bslnVal = baselineData[trl][chx][0, foiPeak]
                ssvepVal = ssvepData[trl][chx][0, foiPeak]
                snrSSVEP[trl][chx] = round(ssvepVal/bslnVal, 3)

        ind = [x for x in range(1, self.numTrials+1)]
        cols = [x for x in self.chNames]
        snr_vep = pd.DataFrame(snrSSVEP, index=ind, columns=cols)
        snr_vep['Average'] = np.around(np.average(snrSSVEP, axis=1), 3)

        snr_vep.to_csv("Results/SSVEP_SNR.txt", sep='\t', index_label="Trial")
    
    def processingPipeline(self):
        ssvepFilter = self.preprocessing(self.ssvepData)
        baselineFilter = self.preprocessing(self.baselineData)

        ssvepPSD = self.psd(ssvepFilter, "SSVEP")
        baselinePSD = self.psd(baselineFilter, "Baseline", graph=False)

        self.cca(ssvepFilter)
        self.erp(ssvepFilter)
        self.snr(ssvepPSD, baselinePSD)


class BackendSocket(WebsocketConsumer):
    def connect(self):
        self.room_group_name = 'test'

        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name,
            self.channel_name
        )

        self.accept()

        self.trial = 0
        self.data = Processing()

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        if (message == 'Ready!'):
            async_to_sync(self.channel_layer.group_send)(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': 'Ready to Begin!'
                }
            )

            self.data.begin()

        elif (message[0:6] == 'Start!'):
            trial = "Trial " + str(self.trial+1)
            self.data.timing(int(message[6:]), cond='start')

            async_to_sync(self.channel_layer.group_send)(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': trial
                }
            )
        elif (message[0:5] == 'Done!'):
            self.data.timing(int(message[5:]), cond='end')
            self.trial += 1
        elif (message[0:9] == 'Complete!'):
            self.data.timing(int(message[9:]), cond='end', lastTrial=True)

            async_to_sync(self.channel_layer.group_send)(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': 'Overt Assessment Complete!'
                }
            )

    def chat_message(self, event):
        message = event['message']

        self.send(text_data=json.dumps({
            'type': 'chat',
            'message': message
        }))

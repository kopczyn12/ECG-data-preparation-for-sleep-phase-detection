import mne
import matplotlib.pyplot as plt
import numpy as np
import biosppy
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values

PATH_EDF = '/home/mkopcz/Desktop/sleep-phases-detection/sleep-phases-detection/haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings/SN134.edf'

def loading_ecg_data(file):
    with open(file) as f:
        data = mne.io.read_raw_edf(file)
    return data

def process_data(data):

    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    ecg_channel_idx = 7
    ecg_channel_name = channels[ecg_channel_idx]
    ecg_data = raw_data[ecg_channel_idx]
    sampling_rate = info['sfreq']

    time_samples = int(0.5* 60 * sampling_rate)
    ecg_data = ecg_data[:time_samples]

    # Process the ECG data and detect R-peaks
    out = biosppy.signals.ecg.ecg(ecg_data, sampling_rate=sampling_rate, show=False)

    # Get R-R intervals
    rr_intervals_list = np.diff(out['rpeaks'])
    # Preprocess R-R intervals
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list, low_rri=300, high_rri=2000)
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method="linear")
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)

    print(f"ECG DATA:\n {ecg_data}")
    print(f"DETECTED R PEAKS:\n {out}")
    print(f"RR_INTERVALS:\n {rr_intervals_list}")
    print(f"RR_INTERVALS_WITHOUT_OUTLIERS:\n {rr_intervals_without_outliers}")
    print(f"INTERPOLATED_RR:\n {interpolated_rr_intervals}")
    print(f"NN_INTERVALS:\n {nn_intervals_list}")
    print(f"INTERPOLATED_NN:\n {interpolated_nn_intervals}")

    return ecg_data, out, rr_intervals_list, rr_intervals_without_outliers, interpolated_rr_intervals, nn_intervals_list, interpolated_nn_intervals

def plot_ecg_and_r_peaks(ecg_data, out):
    plt.figure()
    plt.plot(ecg_data, label = 'ECG Signal')
    plt.plot(out['rpeaks'], ecg_data[out['rpeaks']], 'ro', label='R-peaks')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with R-peaks')
    plt.legend()
    plt.show()



def main():
    data = loading_ecg_data(PATH_EDF)
    ecg_data, out, rr_intervals_list, rr_intervals_without_outliers, interpolated_rr_intervals, nn_intervals_list, interpolated_nn_intervals = process_data(data)
    plot_ecg_and_r_peaks(ecg_data, out)


if __name__ == '__main__':
    main()






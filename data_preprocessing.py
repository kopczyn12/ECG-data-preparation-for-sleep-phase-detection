import mne
import matplotlib.pyplot as plt
import numpy as np
import biosppy
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features
import json
import re
import pandas as pd

PATH_EDF = 'haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings/SN134.edf'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def extract_patient_index(path):
    match = re.search(r'SN\d+', path)
    if match:
        return match.group(0)
    else:
        return None


patient_index = extract_patient_index(PATH_EDF)


def loading_ecg_data(file):
    data = mne.io.read_raw_edf(file, preload=True)
    return data


def process_data(data):
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    ecg_channel_idx = 7
    ecg_channel_name = channels[ecg_channel_idx]
    ecg_data = raw_data[ecg_channel_idx]
    sampling_rate = info['sfreq']
    #
    # time_samples = int(5 * 60 * sampling_rate)
    # ecg_data = ecg_data[:time_samples]

    # Process the ECG data and detect R-peaks
    out = biosppy.signals.ecg.ecg(ecg_data, sampling_rate=sampling_rate, show=False)

    # Get R-R intervals
    rr_intervals_list = np.diff(out['rpeaks'])
    # Preprocess R-R intervals
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list, low_rri=300, high_rri=2000)
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                       interpolation_method="linear")
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)

    return ecg_data, out, rr_intervals_list, rr_intervals_without_outliers, interpolated_rr_intervals, nn_intervals_list, interpolated_nn_intervals


def plot_ecg_and_r_peaks(ecg_data, out):
    plt.figure()
    plt.plot(ecg_data, label='ECG Signal')
    plt.plot(out['rpeaks'], ecg_data[out['rpeaks']], 'ro', label='R-peaks')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with R-peaks')
    plt.legend()
    plt.show()


def hrv_analysis(interpolated_nn_intervals, window_duration=270, step_size=30, patient_index='SN134'):
    num_intervals = len(interpolated_nn_intervals)
    epoch_intervals = int(window_duration)
    step_intervals = int(step_size)

    for i in range(0, num_intervals - epoch_intervals + step_intervals, step_intervals):
        epoch_nn_intervals = interpolated_nn_intervals[i:i + epoch_intervals]
        hrv_indices = get_time_domain_features(epoch_nn_intervals)
        hrv_indices['patient_index'] = patient_index
        # print(f"Epoch {i // step_intervals + 1}:")
        # print(hrv_indices)

        output_file = f'hrv_analysis_{patient_index}.json'
        with open(output_file, 'w') as f:
            json.dump(hrv_indices, f, cls=NumpyEncoder, sort_keys=True, indent=4)


def main():
    data = loading_ecg_data(PATH_EDF)
    ecg_data, out, rr_intervals_list, rr_intervals_without_outliers, interpolated_rr_intervals, nn_intervals_list, interpolated_nn_intervals = process_data(
        data)
    patient_index = extract_patient_index(PATH_EDF)
    hrv_analysis(interpolated_nn_intervals, patient_index=patient_index)

    with open('hrv_analysis_SN134.json', 'r') as f:
        data_dict = json.load(f)
    df = pd.DataFrame.from_dict(data_dict, orient="index").transpose()

    # Set the index of the DataFrame to the patient index
    df['patient_index'] = patient_index
    df.set_index('patient_index', inplace=True)
    print(df)


if __name__ == '__main__':
    main()

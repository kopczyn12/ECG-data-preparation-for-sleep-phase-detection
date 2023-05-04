import mne
import matplotlib.pyplot as plt
import numpy as np
import biosppy
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features, get_frequency_domain_features, get_geometrical_features, get_poincare_plot_features, get_csi_cvi_features
import nolds
import re
import pandas as pd
import os

def extract_patient_index(path):
    match = re.search(r'SN\d+', path)
    if match:
        return match.group(0)
    else:
        return None

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



def hrv_analysis(interpolated_nn_intervals, window_duration=270, step_size=30, patient_index=''):
    num_intervals = len(interpolated_nn_intervals)
    epoch_intervals = int(window_duration)
    step_intervals = int(step_size)
    hrv_indices = []

    for i in range(0, num_intervals - epoch_intervals + step_intervals, step_intervals):
        epoch_nn_intervals = interpolated_nn_intervals[i:i + epoch_intervals]
        if len(epoch_nn_intervals) == 0:
            print(f"Empty nn_intervals for epoch {i}-{i + epoch_intervals}")
            continue
        try:
            time_domain_features = get_time_domain_features(epoch_nn_intervals)
            frequency_domain_features = get_frequency_domain_features(epoch_nn_intervals)
            poincare_plot_features = get_poincare_plot_features(epoch_nn_intervals)
            csi_cvi_features = get_csi_cvi_features(epoch_nn_intervals)
            epoch_hrv_indices = {}
            epoch_hrv_indices.update(time_domain_features)
            epoch_hrv_indices.update(frequency_domain_features)
            epoch_hrv_indices.update(poincare_plot_features)
            epoch_hrv_indices.update(csi_cvi_features)
            # calculate asymmetry of Poincare plot area index
            left_area = poincare_plot_features['sd1'] ** 2
            right_area = poincare_plot_features['sd2'] ** 2
            total_area = left_area + right_area
            asymmetry_index = abs(left_area - right_area) / total_area
            epoch_hrv_indices['C1d'] = asymmetry_index
            epoch_hrv_indices['C2d'] = csi_cvi_features['cvi']
            epoch_hrv_indices['C2a'] = csi_cvi_features['csi']
            hf_power = frequency_domain_features['hf']
            short_term_var_of_accelerations = hf_power / len(epoch_nn_intervals)
            epoch_hrv_indices['SD1a'] = short_term_var_of_accelerations
            epoch_nn_intervals = np.array(epoch_nn_intervals)
            deceleration_index = (epoch_nn_intervals[1:] - epoch_nn_intervals[:-1] > 50).sum() / len(epoch_nn_intervals)
            acceleration_index = (epoch_nn_intervals[1:] - epoch_nn_intervals[:-1] < -50).sum() / len(
                epoch_nn_intervals)

            total_contributions_d = np.sqrt(deceleration_index) * csi_cvi_features['cvi']
            total_contributions_a = np.sqrt(acceleration_index) * csi_cvi_features['csi']

            epoch_hrv_indices['Cd'] = total_contributions_d
            epoch_hrv_indices['Ca'] = total_contributions_a
            epoch_hrv_indices['SD2d'] = np.var(epoch_hrv_indices['Cd'])
            epoch_hrv_indices['SD2a'] = np.var(epoch_hrv_indices['Ca'])
            # calculate percentage of inflection points of RR intervals series
            rr_diff = np.diff(epoch_nn_intervals)
            num_inflection_points = ((rr_diff[1:] > 0) & (rr_diff[:-1] < 0)).sum()
            percentage_inflection_points = num_inflection_points / len(epoch_nn_intervals)
            epoch_hrv_indices['PIP'] = percentage_inflection_points

            # calculate MCVNN index
            median_abs_dev = np.median(np.abs(epoch_nn_intervals - np.median(epoch_nn_intervals)))
            median_abs_diff = np.median(np.abs(np.diff(epoch_nn_intervals)))
            mcvnn = median_abs_dev / median_abs_diff
            epoch_hrv_indices['MCVNN'] = mcvnn

            # Calculate DFA_alpha1
            dfa_alpha1 = nolds.dfa(epoch_nn_intervals, nvals=None, overlap=True, order=1, fit_trend="poly", fit_exp="RANSAC", debug_plot=False, debug_data=False, plot_file=None)
            epoch_hrv_indices['DFA_alpha1'] = dfa_alpha1

            sd2d = np.var(total_contributions_d)
            sd2a = np.var(total_contributions_a)
            epoch_hrv_indices['SD2d'] = sd2d
            epoch_hrv_indices['SD2a'] = sd2a
            epoch_hrv_indices['sdnn/cvnni'] = epoch_hrv_indices['sdnn'] / epoch_hrv_indices['cvnni']
            epoch_hrv_indices['patient_index'] = patient_index
            epoch_hrv_indices['epoch_index'] = i // step_intervals + 1
            hrv_indices.append(epoch_hrv_indices)
        except ValueError as e:
            print(f"Error calculating features for epoch {i}-{i+epoch_intervals}: {str(e)}")
            continue

    return pd.DataFrame(hrv_indices)

def main():

    edf_paths = []
    for i in range(12, 155):
        if i < 100:
            file_name = f"SN0{i}.edf"
            file_path = os.path.join("haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings/", file_name)
            if os.path.isfile(file_path):
                edf_paths.append(file_path)
        else:
            file_name = f"SN{i}.edf"
            file_path = os.path.join("haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings/", file_name)
            if os.path.isfile(file_path):
                edf_paths.append(file_path)

    df_data = []
    for i in range(len(edf_paths)):
        data = loading_ecg_data(edf_paths[i])
        ecg_data, out, rr_intervals_list, rr_intervals_without_outliers, interpolated_rr_intervals, nn_intervals_list, interpolated_nn_intervals = process_data(data)
        patient_index = extract_patient_index(edf_paths[i])
        hrv_indices = hrv_analysis(interpolated_nn_intervals, window_duration=270, step_size=30, patient_index=patient_index)
        if hrv_indices is not None:
            df_data.append(hrv_indices)

    result = pd.concat(df_data)
    result.set_index("patient_index", inplace=True)
    result_final = result.drop(['sdnn', 'cvnni', 'cvi', 'nni_50', 'nni_20', 'pnni_20', 'rmssd', 'range_nni',
                          'cvsd', 'mean_hr', 'max_hr', 'min_hr', 'std_hr', 'lf'
                          , 'hf', 'lf_hf_ratio', 'hfnu', 'total_power', 'vlf', 'sd1', 'sd2', 'ratio_sd2_sd1',
                           'Modified_csi'], axis=1)
    result_final.to_csv('features.csv')

if __name__ == '__main__':
    main()

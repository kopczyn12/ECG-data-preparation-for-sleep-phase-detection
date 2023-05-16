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
   
    out = biosppy.signals.ecg.ecg(ecg_data, sampling_rate=sampling_rate, show=False)
    rr_intervals_list = np.diff(out["ts"][out['rpeaks']]) * 1000 
    
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
    hr_values = []

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
            
            # separate csv for hr
            epoch_hr = {}
            epoch_hr['mean_hr'] = time_domain_features['mean_hr']
            epoch_hr['patient_index'] = patient_index
            epoch_hr['epoch_index'] = i // step_intervals + 1
            hr_values.append(epoch_hr)

            #separate for hrv analysis
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


    return pd.DataFrame(hrv_indices), pd.DataFrame(hr_values)

def concatenate_sleep_scoring_files(directory_path):
    data_frames = []  # list to hold all data frames
    filenames = os.listdir(directory_path)
    filenames.sort()  # sort the filenames in ascending order

    annotation_set = False
    for filename in filenames:
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            patient_index = filename.split('_')[0]  # get the patient index from the filename

            df = pd.read_csv(file_path, sep=',', header=None, usecols=[4], skiprows=1, names=['Annotation'])
            df['patient_index'] = patient_index  # add the patient index column

            if not annotation_set:
                annotation_set = True
                data_frames.append(df[['Annotation', 'patient_index']])
            else:
                data_frames.append(df[['Annotation', 'patient_index']].iloc[1:])

    concatenated_df = pd.concat(data_frames)  # concatenate all data frames
    concatenated_df['Annotation'] = concatenated_df['Annotation'].str.replace('Sleep stage ', '') # remove 'Sleep stage' from Annotation column values
    concatenated_df.set_index("patient_index", inplace=True)  # set the patient_index column as the index
    return concatenated_df

def calculate_stats(dataframe, columns, output_csv):
    selected_data = dataframe[columns]
    
    stats_dict = {
        'Column': [],
        'Mean': [],
        'Max': [],
        'Min': [],
        'Std Dev': [],
        'Variance': []
    }
    
    for column in columns:
        column_data = selected_data[column]
        
        mean = column_data.mean()
        maximum = column_data.max()
        minimum = column_data.min()
        std_dev = column_data.std()
        variance = column_data.var()
        
        stats_dict['Column'].append(column)
        stats_dict['Mean'].append(mean)
        stats_dict['Max'].append(maximum)
        stats_dict['Min'].append(minimum)
        stats_dict['Std Dev'].append(std_dev)
        stats_dict['Variance'].append(variance)
        
    stats_df = pd.DataFrame(stats_dict)
    
    stats_df.to_csv(output_csv, index=False) 
    
    return stats_df

def main():

    # #files handling
    edf_paths = []
    for i in range(1, 155):
        if i < 10:
            file_name = f"SN00{i}.edf"
            file_path = os.path.join("haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings/", file_name)
            if os.path.isfile(file_path):
                edf_paths.append(file_path)

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
    hr_data = []
    for i in range(len(edf_paths)):
        data = loading_ecg_data(edf_paths[i])
        ecg_data, out, rr_intervals_list, rr_intervals_without_outliers, interpolated_rr_intervals, nn_intervals_list, interpolated_nn_intervals = process_data(data)
        patient_index = extract_patient_index(edf_paths[i])
        hrv_indices, heart_rate_value = hrv_analysis(interpolated_nn_intervals, window_duration=270, step_size=30, patient_index=patient_index)

        if heart_rate_value is not None:
            hr_data.append(heart_rate_value)
        if hrv_indices is not None:
            df_data.append(hrv_indices)

    results_hr = pd.concat(hr_data)
    results_hr.set_index("patient_index", inplace=True)
    results_hr.to_csv('results_hr.csv')

    result = pd.concat(df_data)
    result.set_index("patient_index", inplace=True)
    result_final = result.drop(['sdnn', 'cvnni', 'cvi', 'nni_50', 'nni_20', 'pnni_20', 'rmssd', 'range_nni',
                          'cvsd','std_hr', 'lf'
                          , 'hf', 'lf_hf_ratio', 'hfnu', 'total_power', 'vlf', 'sd1', 'sd2', 'ratio_sd2_sd1',
                           'Modified_csi'], axis=1)

    result_final.to_csv('features.csv')
    annotations = concatenate_sleep_scoring_files('haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings/')
    annotations.to_csv('annotations.csv')

    df = pd.read_csv('features.csv')
    output_file = 'statistics_for_whole_population.csv'
    result = calculate_stats(df, ['mean_nni', 'mean_hr', 'max_hr', 'min_hr'], output_file)
    
if __name__ == '__main__':
    main()

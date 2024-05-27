import mne
import matplotlib.pyplot as plt
import numpy as np
import biosppy
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features, get_frequency_domain_features, get_geometrical_features, get_poincare_plot_features, get_csi_cvi_features
import nolds
import re
import pandas as pd
import os
from mne.io import Raw
from typing import Tuple, List, Any, Optional, Dict
from omegaconf import DictConfig, OmegaConf
from utils.logger import initialize_logger, logger

def extract_patient_index(path: str) -> Optional[str]:
    """
    Extracts the patient index from a given path.

    This function searches the path for a pattern that matches 'SN' followed by one or more digits.
    If a match is found, it returns the matched string; otherwise, it returns None.

    Args:
        path (str): The file path that contains the patient index.

    Returns:
    Optional[str]: The extracted patient index as a string if found, otherwise None.
    """
    match = re.search(r'SN\d+', path)
    if match:
        return match.group(0)
    else:
        return None


def loading_ecg_data(file: str) -> Raw:
    """
    Loads ECG data from a specified file using the MNE library.

    This function reads an EDF file containing ECG data and loads it into memory. It uses
    the `read_raw_edf` method from the MNE library, enabling the `preload` parameter to
    load all data at once into memory for further processing.

    Args:
        file (str): The path to the EDF file containing the ECG data.

    Returns:
    Raw: An MNE Raw object containing the loaded ECG data.
    """
    data = mne.io.read_raw_edf(file, preload=True)
    return data


def process_data(cfg: DictConfig, data: Any) -> Tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes ECG data to extract and preprocess R-R intervals.

    Args:
        cfg (DictConfig): Configuration for the data processing.
        data (Any): An object containing ECG data, metadata, and channel names.

    Returns:
        Tuple containing:
              ecg_data (np.ndarray): The raw ECG data from the specified channel.
              out (dict): The output dictionary from biosppy containing ECG analysis results.
              interpolated_nn_intervals (np.ndarray): NN intervals after interpolation of NaN values and processing.
    """

    # Extract raw data, metadata, and channel names
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names

    # Define the index and name for the ECG channel
    ecg_channel_idx = 7
    ecg_channel_name = channels[ecg_channel_idx]

    # Extract ECG data for the specified channel
    ecg_data = raw_data[ecg_channel_idx]

    # Get the sampling rate from the metadata
    sampling_rate = info['sfreq']
    logger.info(f"Sampling rate: {sampling_rate}")

    # Perform ECG analysis using biosppy
    out = biosppy.signals.ecg.ecg(ecg_data, sampling_rate=sampling_rate, show=False)

    # Compute R-R intervals in milliseconds
    rr_intervals_list = np.diff(out["ts"][out['rpeaks']]) * 1000

    # Preprocess R-R intervals by removing outliers
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list, low_rri=cfg.pipeline.analysis.low_rri, high_rri=cfg.pipeline.analysis.high_rri)

    # Interpolate NaN values in the R-R intervals
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method=cfg.pipeline.analysis.interpolation_method)

    # Remove ectopic beats from the R-R intervals to get NN intervals
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method=cfg.pipeline.analysis.ectopic_beat_removal_method)

    # Interpolate NaN values in the NN intervals
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)

    return ecg_data, out, interpolated_nn_intervals

def plot_ecg_and_r_peaks(ecg_data: np.ndarray, out: Dict[str, Any], save_dir: str, file_name: str, segment_length: int = 10000, sampling_rate: int = 256) -> None:
    """
    Plots a segment of the ECG signal with detected R-peaks and saves the plot to the specified directory.

    Args:
        ecg_data (np.ndarray): The raw ECG data.
        out (Dict[str, Any]): Output dictionary from biosppy containing ECG analysis results, including R-peaks.
        save_dir (str): Directory to save the plot.
        file_name (str): Name of the file to save the plot as.
        segment_length (int, optional): Length of the ECG segment to plot. Defaults to 10000 samples.
        sampling_rate (int, optional): Sampling rate of the ECG data in Hz. Defaults to 256 Hz.

    Returns:
        None
    """
    # Determine the end index for the segment
    end_index = min(segment_length, len(ecg_data))

    # Select the segment of ECG data and corresponding R-peaks within the segment
    ecg_segment = ecg_data[:end_index]
    rpeaks_segment = out['rpeaks'][out['rpeaks'] < end_index]

    # Convert sample indices to time in seconds
    time_segment = np.arange(0, end_index) / sampling_rate
    rpeaks_time = rpeaks_segment / sampling_rate

    # Create a new figure for the plot
    plt.figure(figsize=(10, 5))

    # Plot the segment of the raw ECG signal
    plt.plot(time_segment, ecg_segment, label='ECG Signal')

    # Plot the detected R-peaks on the ECG segment
    plt.plot(rpeaks_time, ecg_segment[rpeaks_segment], 'ro', label='R-peaks')

    # Label the x-axis as 'Time (s)'
    plt.xlabel('Time (s)')

    # Label the y-axis as 'Amplitude'
    plt.ylabel('Amplitude')

    # Set the title of the plot
    plt.title('Segment of ECG Signal with R-peaks')

    # Display the legend for the plot
    plt.legend()

    # Save the plot to the specified directory with the provided file name
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)

    # Close the plot to free up memory
    plt.close()

def hrv_analysis(interpolated_nn_intervals: np.ndarray,
                 window_duration: int = 270,
                 step_size: int = 30,
                 patient_index: str = '') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs HRV (Heart Rate Variability) analysis on interpolated NN intervals.

    Args:
        interpolated_nn_intervals (np.ndarray): Array of interpolated NN intervals.
        window_duration (int, optional): Duration of each window for analysis in seconds. Defaults to 270.
        step_size (int, optional): Step size for sliding window in seconds. Defaults to 30.
        patient_index (str, optional): Identifier for the patient. Defaults to an empty string.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing HRV indices and heart rate values respectively.
    """

    num_intervals = len(interpolated_nn_intervals)
    epoch_intervals = int(window_duration)
    step_intervals = int(step_size)
    hrv_indices = []
    hr_values = []

    # Iterate over the NN intervals with a sliding window approach
    for i in range(0, num_intervals - epoch_intervals + step_intervals, step_intervals):
        epoch_nn_intervals = interpolated_nn_intervals[i:i + epoch_intervals]

        # Check if the epoch contains any NN intervals
        if len(epoch_nn_intervals) == 0:
            print(f"Empty nn_intervals for epoch {i}-{i + epoch_intervals}")
            continue

        try:
            # Convert the list to a NumPy array for numerical operations
            epoch_nn_intervals = np.array(epoch_nn_intervals)

            # Calculate various HRV features for the current epoch
            time_domain_features = get_time_domain_features(epoch_nn_intervals)
            frequency_domain_features = get_frequency_domain_features(epoch_nn_intervals)
            poincare_plot_features = get_poincare_plot_features(epoch_nn_intervals)
            csi_cvi_features = get_csi_cvi_features(epoch_nn_intervals)

            # Store heart rate information in a separate dictionary
            epoch_hr = {
                'mean_hr': time_domain_features['mean_hr'],
                'patient_index': patient_index,
                'epoch_index': i // step_intervals + 1
            }
            hr_values.append(epoch_hr)

            # Combine all HRV features into one dictionary
            epoch_hrv_indices = {}
            epoch_hrv_indices.update(time_domain_features)
            epoch_hrv_indices.update(frequency_domain_features)
            epoch_hrv_indices.update(poincare_plot_features)
            epoch_hrv_indices.update(csi_cvi_features)

            # Calculate additional HRV indices and add them to the dictionary
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

            # Calculate deceleration and acceleration indices
            deceleration_index = ((epoch_nn_intervals[1:] - epoch_nn_intervals[:-1]) > 50).sum() / len(epoch_nn_intervals)
            acceleration_index = ((epoch_nn_intervals[1:] - epoch_nn_intervals[:-1]) < -50).sum() / len(epoch_nn_intervals)
            total_contributions_d = np.sqrt(deceleration_index) * csi_cvi_features['cvi']
            total_contributions_a = np.sqrt(acceleration_index) * csi_cvi_features['csi']
            epoch_hrv_indices['Cd'] = total_contributions_d
            epoch_hrv_indices['Ca'] = total_contributions_a
            epoch_hrv_indices['SD2d'] = np.var(epoch_hrv_indices['Cd'])
            epoch_hrv_indices['SD2a'] = np.var(epoch_hrv_indices['Ca'])

            # Calculate percentage of inflection points in RR intervals
            rr_diff = np.diff(epoch_nn_intervals)
            num_inflection_points = ((rr_diff[1:] > 0) & (rr_diff[:-1] < 0)).sum()
            percentage_inflection_points = num_inflection_points / len(epoch_nn_intervals)
            epoch_hrv_indices['PIP'] = percentage_inflection_points

            # Calculate MCVNN index
            median_abs_dev = np.median(np.abs(epoch_nn_intervals - np.median(epoch_nn_intervals)))
            median_abs_diff = np.median(np.abs(np.diff(epoch_nn_intervals)))
            mcvnn = median_abs_dev / median_abs_diff
            epoch_hrv_indices['MCVNN'] = mcvnn

            # Calculate DFA_alpha1 using detrended fluctuation analysis
            dfa_alpha1 = nolds.dfa(epoch_nn_intervals, nvals=None, overlap=True, order=1, fit_trend="poly", fit_exp="RANSAC")
            epoch_hrv_indices['DFA_alpha1'] = dfa_alpha1

            # Calculate additional SD2 indices
            sd2d = np.var(total_contributions_d)
            sd2a = np.var(total_contributions_a)
            epoch_hrv_indices['SD2d'] = sd2d
            epoch_hrv_indices['SD2a'] = sd2a
            epoch_hrv_indices['sdnn/cvnni'] = epoch_hrv_indices['sdnn'] / epoch_hrv_indices['cvnni']
            epoch_hrv_indices['patient_index'] = patient_index
            epoch_hrv_indices['epoch_index'] = i // step_intervals + 1

            # Append the HRV indices for the current epoch to the list
            hrv_indices.append(epoch_hrv_indices)

        except ValueError as e:
            print(f"Error calculating features for epoch {i}-{i + epoch_intervals}: {str(e)}")
            continue

    # Convert the list of dictionaries to DataFrames and return
    return pd.DataFrame(hrv_indices), pd.DataFrame(hr_values)


def concatenate_sleep_scoring_files(directory_path: str) -> pd.DataFrame:
    """
    Concatenates sleep scoring files from a directory into a single DataFrame.

    Args:
        directory_path (str): Path to the directory containing sleep scoring files.

    Returns:
        pd.DataFrame: A DataFrame containing concatenated sleep scoring data with patient indices.
    """

    data_frames = []  # List to hold all data frames
    filenames = os.listdir(directory_path)
    filenames.sort()  # Sort the filenames in ascending order

    annotation_set = False
    for filename in filenames:
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            patient_index = filename.split('_')[0]  # Get the patient index from the filename

            # Read the file into a DataFrame
            df = pd.read_csv(file_path, sep=',', header=None, usecols=[4], skiprows=1, names=['Annotation'])
            df['patient_index'] = patient_index  # Add the patient index column

            # Add DataFrame to the list, handling the first file separately to avoid duplicate header
            if not annotation_set:
                annotation_set = True
                data_frames.append(df[['Annotation', 'patient_index']])
            else:
                data_frames.append(df[['Annotation', 'patient_index']].iloc[1:])

    # Concatenate all data frames into one
    concatenated_df = pd.concat(data_frames)

    # Remove 'Sleep stage' from Annotation column values
    concatenated_df['Annotation'] = concatenated_df['Annotation'].str.replace('Sleep stage ', '')

    # Set the patient_index column as the index
    concatenated_df.set_index("patient_index", inplace=True)

    return concatenated_df

def calculate_stats(dataframe: pd.DataFrame, columns: List[str], output_csv: str) -> pd.DataFrame:
    """
    Calculates statistical metrics for selected columns in a DataFrame and saves the results to a CSV file.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data.
        columns (List[str]): A list of column names to calculate statistics for.
        output_csv (str): The path to the output CSV file where the statistics will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics.
    """
    # Select specified columns from the DataFrame
    selected_data = dataframe[columns]

    # Dictionary to hold the statistics for each column
    stats_dict = {
        'Column': [],
        'Mean': [],
        'Max': [],
        'Min': [],
        'Std Dev': [],
        'Variance': []
    }

    # Calculate statistics for each specified column
    for column in columns:
        column_data = selected_data[column]

        mean = column_data.mean()
        maximum = column_data.max()
        minimum = column_data.min()
        std_dev = column_data.std()
        variance = column_data.var()

        # Append the statistics to the dictionary
        stats_dict['Column'].append(column)
        stats_dict['Mean'].append(mean)
        stats_dict['Max'].append(maximum)
        stats_dict['Min'].append(minimum)
        stats_dict['Std Dev'].append(std_dev)
        stats_dict['Variance'].append(variance)

    # Convert the dictionary to a DataFrame
    stats_df = pd.DataFrame(stats_dict)

    # Save the statistics DataFrame to a CSV file
    stats_df.to_csv(output_csv, index=False)

    return stats_df

def process_sleep_scoring_files(directory_path: str) -> pd.DataFrame:
    """
    Processes sleep scoring files from a specified directory and concatenates them into a single DataFrame.

    Args:
        directory_path (str): Path to the directory containing sleep scoring files.

    Returns:
        pd.DataFrame: A DataFrame containing concatenated sleep scoring data with patient indices.
    """
    data_frames = []  # List to hold all data frames
    filenames = os.listdir(directory_path)
    filenames.sort()  # Sort the filenames in ascending order

    annotation_set = False
    for filename in filenames:
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            patient_index = filename.split('_')[0]  # Get the patient index from the filename

            # Read the file into a DataFrame
            df = pd.read_csv(file_path, sep=',', header=None, usecols=[4], skiprows=1, names=['Annotation'])
            df['patient_index'] = patient_index  # Add the patient index column

            # Add DataFrame to the list, handling the first file separately to avoid duplicate header
            if not annotation_set:
                annotation_set = True
                data_frames.append(df[['Annotation', 'patient_index']])
            else:
                data_frames.append(df[['Annotation', 'patient_index']].iloc[1:])

    # Concatenate all data frames into one
    concatenated_df = pd.concat(data_frames)

    # Remove 'Sleep stage' from Annotation column values
    concatenated_df['Annotation'] = concatenated_df['Annotation'].str.replace('Sleep stage ', '')

    # Set the patient_index column as the index
    concatenated_df.set_index("patient_index", inplace=True)

    return concatenated_df

def dataset_preparation(cfg: DictConfig) -> None:
    """
    Main function to process ECG and sleep scoring data, extract HRV features, and save the results to CSV files.

    Args:
        cfg (DictConfig): The configuration for the dataset preparation.
    Returns:
        None
    """

    # List to store paths of EDF files
    edf_paths = []
    logger.info("Collecting EDF file paths...")
    # Collect EDF file paths
    for i in range(1, 155):
        if i < 10:
            file_name = f"SN00{i}.edf"
        elif i < 100:
            file_name = f"SN0{i}.edf"
        else:
            file_name = f"SN{i}.edf"
        logger.info(f"Checking file: {file_name}")
        file_path = os.path.join(cfg.pipeline.directories.edf_directory, file_name)
        if os.path.isfile(file_path):
            edf_paths.append(file_path)
    df_data = []
    hr_data = []
    logger.info("Processing EDF files...")
    # Process each EDF file
    for file_path in edf_paths:
        logger.info(f"Processing file: {file_path}")
        data = loading_ecg_data(file_path)
        logger.info(f"Loaded ECG data from {file_path}")
        ecg_data, out, interpolated_nn_intervals = process_data(cfg, data)
        logger.info(f"Processed ECG data from {file_path}")
        patient_index = extract_patient_index(file_path)
        logger.info(f"Extracted patient index: {patient_index}")
        if cfg.pipeline.plotting.plot_ecg:
            if not os.path.exists(cfg.pipeline.directories.plots_directory):
                os.makedirs(cfg.pipeline.directories.plots_directory, exist_ok=True)
            logger.info("Plotting ECG signal with R-peaks...")
            plot_ecg_and_r_peaks(ecg_data, out, cfg.pipeline.directories.plots_directory, f"{patient_index}_ecg_plot.png", segment_length=cfg.pipeline.plotting.segment_length, sampling_rate=cfg.pipeline.plotting.sampling_rate)
            logger.info(f"Plot saved at {cfg.pipeline.directories.plots_directory}")
        logger.info("Performing HRV analysis...")
        hrv_indices, heart_rate_value = hrv_analysis(interpolated_nn_intervals, window_duration=cfg.pipeline.analysis.window_duration, step_size=cfg.pipeline.analysis.step_size, patient_index=patient_index)
        logger.info("HRV analysis completed.")

        if heart_rate_value is not None:
            hr_data.append(heart_rate_value)
        if hrv_indices is not None:
            df_data.append(hrv_indices)
    logger.info("Saving results to CSV files...")
    # Save heart rate results
    results_hr = pd.concat(hr_data)
    results_hr.set_index("patient_index", inplace=True)
    results_hr.to_csv(cfg.pipeline.directories.output_hr_file)
    logger.info(f"Saved heart rate results to {cfg.pipeline.directories.output_hr_file}")

    # Save HRV feature results
    result = pd.concat(df_data)
    result.set_index("patient_index", inplace=True)
    result_final = result.drop([
        'sdnn', 'cvnni', 'cvi', 'nni_50', 'nni_20', 'pnni_20', 'rmssd',
        'range_nni', 'cvsd', 'std_hr', 'lf', 'hf', 'lf_hf_ratio', 'hfnu',
        'total_power', 'vlf', 'sd1', 'sd2', 'ratio_sd2_sd1', 'Modified_csi'
    ], axis=1)
    result_final.to_csv(cfg.pipeline.directories.output_features_file)
    logger.info(f"Saved HRV feature results to {cfg.pipeline.directories.output_features_file}")

    # Process and save annotations
    annotations = process_sleep_scoring_files(cfg.pipeline.directories.data_directory)
    annotations.to_csv(cfg.pipeline.directories.output_annotations_file)
    logger.info(f"Saved sleep scoring annotations to {cfg.pipeline.directories.output_annotations_file}")

    # Concatenate annotations and features to achieve final dataset
    # Define valid annotations
    valid_annotations = [' W', ' N1', ' N2', ' N3', ' R']

    # Join the dataframes
    dataset = result_final.join(annotations)

    # Filter the dataset to keep only rows with valid annotations
    dataset_filtered = dataset[dataset['Annotation'].isin(valid_annotations)]

    # Save the filtered dataset to CSV (assuming cfg.pipeline.directories has been defined and contains the paths)
    dataset_filtered.to_csv(cfg.pipeline.directories.output_dataset_file)

    # Logging the save action
    logger.info(f"Saved filtered dataset to {cfg.pipeline.directories.output_dataset_file}")

    # Calculate and save statistics
    df = pd.read_csv(cfg.pipeline.directories.output_features_file)
    calculate_stats(df, ['mean_nni', 'mean_hr', 'max_hr', 'min_hr'], cfg.pipeline.directories.output_stats_file)
    logger.info(f"Saved statistics to {cfg.pipeline.directories.output_stats_file}")




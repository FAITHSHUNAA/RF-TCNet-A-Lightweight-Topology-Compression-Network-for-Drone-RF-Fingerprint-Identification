import numpy as np
import librosa
import matplotlib.pyplot as plt
import h5py
import os
import glob
from tkinter import filedialog
from tkinter import Tk


# select folder
def select_directory():
    root = Tk()
    root.withdraw()  # Do not display the main window
    folder_path = filedialog.askdirectory(title="Select the folder where the .mat file is located")
    return folder_path


# Select multiple .mat files
def select_mat_files(folder_path):
    mat_files = glob.glob(os.path.join(folder_path, '*.mat'))
    selected_files = filedialog.askopenfilenames(
        title="Select one or more .mat files",
        initialdir=folder_path,
        filetypes=[("MAT Files", "*.mat")]
    )
    return selected_files


def process_mat_files(mat_files, output_base_folder):
    # Preprocessing: Calculate the global maximum amplitude and dB range
    global_max_amplitude = -np.inf
    for mat_file in mat_files:
        data = h5py.File(mat_file, 'r')
        RF0_I = data['RF0_I'][0]
        global_max_amplitude = max(global_max_amplitude, np.max(np.abs(RF0_I)))

    # Calculate the global dB range
    global_min_dB = +np.inf
    global_max_dB = -np.inf
    for mat_file in mat_files:
        data = h5py.File(mat_file, 'r')
        RF0_I = data['RF0_I'][0]
        frame_size = 3_000_000
        num_frames = len(RF0_I) // frame_size
        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = (i + 1) * frame_size
            data_ch0 = RF0_I[start_idx:end_idx]
            data_ch0 = np.array(data_ch0, dtype=float)
            stft = librosa.stft(data_ch0, n_fft=1024, hop_length=512, win_length=1024, window='hamming')
            log_spec = librosa.amplitude_to_db(np.abs(stft), ref=global_max_amplitude)
            global_min_dB = min(global_min_dB, np.min(log_spec))
            global_max_dB = max(global_max_dB, np.max(log_spec))

    # Process and save the time-frequency graph
    for mat_file in mat_files:
        data = h5py.File(mat_file, 'r')
        RF0_I = data['RF0_I'][0]
        base_name = os.path.basename(mat_file).split('.')[0]
        output_folder = os.path.join(output_base_folder, base_name)
        os.makedirs(output_folder, exist_ok=True)

        frame_size = 5_000_000
        num_frames = len(RF0_I) // frame_size
        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = (i + 1) * frame_size
            data_ch0 = RF0_I[start_idx:end_idx]
            data_ch0 = np.array(data_ch0, dtype=float)

            stft = librosa.stft(data_ch0, n_fft=1024, hop_length=512, win_length=1024, window='hamming')
            log_spec = librosa.amplitude_to_db(np.abs(stft), ref=global_max_amplitude)

            save_spectrogram(log_spec, output_folder, base_name, i, global_min_dB, global_max_dB)


def save_spectrogram(spectrogram, output_folder, file_name, index, vmin, vmax):
    plt.figure(figsize=(1.5, 1))
    plt.imshow(spectrogram, cmap='jet', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path = os.path.join(output_folder, f'{file_name}_frame_{index}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# main
if __name__ == '__main__':
    # Select the input folder and obtain the .mat file selected by the user
    folder_path = select_directory()
    selected_files = select_mat_files(folder_path)

    # Select output folder
    output_folder = filedialog.askdirectory(title="Select the folder for saving the time-frequency graph")

    # Batch processing of the selected .mat files
    process_mat_files(selected_files, output_folder)

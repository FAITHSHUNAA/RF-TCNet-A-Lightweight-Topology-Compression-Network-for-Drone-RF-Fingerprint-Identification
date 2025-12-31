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
    for mat_file in mat_files:
        print(f"Processing: {os.path.basename(mat_file)}")  

        with h5py.File(mat_file, 'r') as data:
            RF0_I = data['RF0_I'][0]  

            # Step 1: Calculate the maximum amplitude
            file_max_amplitude = np.max(np.abs(RF0_I))

            # Step 2: Statistical dB range
            file_min_dB = np.inf
            file_max_dB = -np.inf

            frame_size_stats = 3_000_000 
            num_frames_stats = len(RF0_I) // frame_size_stats

            for i in range(num_frames_stats):
                start_idx = i * frame_size_stats
                end_idx = (i + 1) * frame_size_stats
                segment = np.array(RF0_I[start_idx:end_idx], dtype=float)

                stft = librosa.stft(segment, n_fft=1024, hop_length=512, win_length=1024, window='hamming')
                log_spec = librosa.amplitude_to_db(np.abs(stft), ref=file_max_amplitude)

                file_min_dB = min(file_min_dB, np.min(log_spec))
                file_max_dB = max(file_max_dB, np.max(log_spec))

            if file_min_dB == np.inf:
                file_min_dB = -80
                file_max_dB = 0

            # Step 3: Generate and save the official score
            base_name = os.path.basename(mat_file).split('.')[0]
            output_folder = os.path.join(output_base_folder, base_name)
            os.makedirs(output_folder, exist_ok=True)

            frame_size_final = 3_000_000 
            num_frames_final = len(RF0_I) // frame_size_final

            for i in range(num_frames_final):
                start_idx = i * frame_size_final
                end_idx = (i + 1) * frame_size_final
                segment = np.array(RF0_I[start_idx:end_idx], dtype=float)

                stft = librosa.stft(segment, n_fft=1024, hop_length=512, win_length=1024, window='hamming')
                log_spec = librosa.amplitude_to_db(np.abs(stft), ref=file_max_amplitude)

                save_spectrogram(log_spec, output_folder, base_name, i, file_min_dB, file_max_dB)


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

    if not selected_files:
        print("No files selected, exiting.")
    else:
        # Select output folder
        output_folder = filedialog.askdirectory(title="Select the folder for saving the time-frequency graph")
        if output_folder:
            # Batch processing of the selected .mat files
            process_mat_files(selected_files, output_folder)
            print("All files processed successfully!")

# my-project
1. Project Introduction (Unmanned Aerial Vehicle RF Fingerprint Recognition Model)
This is a project focused on the classification of drone RF signals. This project involves pre-processing and deep learning models. Feature extraction is a topology compression network specifically designed for the RF signal spectrum of unmanned aerial vehicles, achieving precise classification with extremely low trainable parameters.

2. Dataset description
Public datasets DroneRF and DroneRFa

3. Install dependency methods
pip install -r requirements.txt

4. Usage methods
Firstly, download the public datasets DroneRF and DroneRFa, and then perform ECSG on them to generate spectrograms, thereby obtaining the spectrogram dataset. Then use dataset partitioning code to randomly divide it into training set, validation set, and testing set. Next, train it using RF_TCNet_Train.exe, and finally test it using RF_TCNet_Test. py

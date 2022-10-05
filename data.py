import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
# from config import PEAK_CFG

class PEAK_CFG:
    plateau_size = None
    prominance = 0.0000001
    # prominance = None
    height = 0.0000001
    distance = 100
    threshold = 0.000001


def get_velocity_profile(tst_case, flow_rate_data):
    velocity_profile = flow_rate_data[tst_case]
    return velocity_profile

def detect_peak(x, tst_case_name):
    """
    Detect peak in velocity profile
    
    scipy.signal.find_peaks hyperparameters are:
    height: minimum height of peaks
    distance: minimum distance between peaks
    width: minimum width of peaks
    prominence: minimum prominence of peaks
    plateau_size: minimum plateau size of peaks
    """
    plt.figure()
    peaks, _ = find_peaks(x, plateau_size=PEAK_CFG.plateau_size,
                            prominence=PEAK_CFG.prominance,
                            height=PEAK_CFG.height,
                            distance=PEAK_CFG.distance,
                            threshold=PEAK_CFG.threshold)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.title(tst_case_name)
    plt.show()

if __name__ == "__main__":
    param_data_address = "data/Results_All.csv"
    flow_rate_data_address = "data/Results_Flow rate.csv"
    param_data = pd.read_csv(param_data_address)
    flow_rate_data = pd.read_csv(flow_rate_data_address)
    test_cases = flow_rate_data.columns[1:]
    test_cases = [f'tst-12.{i}' for i in range(1, 11)]
    for tst_case in test_cases:
        velocity_profile = get_velocity_profile(tst_case, flow_rate_data)
        detect_peak(velocity_profile, tst_case)
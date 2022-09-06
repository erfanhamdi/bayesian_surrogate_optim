#%%
from termios import VEOF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
velocity_data = "/Users/venus/Erfan/paper_based_pump/velocity_trend.csv"
velocity_data = pd.read_csv(velocity_data)
# %%
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#%%
test_cases = velocity_data.columns[1:]
#%%
for test_case in test_cases[:1]:
    plt.figure()
    x = velocity_data[test_case]
    peaks, _ = find_peaks(x, plateau_size=0.1)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()
# %%

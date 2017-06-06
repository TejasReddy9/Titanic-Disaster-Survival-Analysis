# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
    plt.show()


sinplot()
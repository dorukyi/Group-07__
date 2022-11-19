import pandas as pd
import numpy as np

gg = pd.DataFrame(columns=["Dzp", "sb"])

g = np.array([53, 34, 654, 35451, 10])

gg["Dzp"] = g

print(gg)

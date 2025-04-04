import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('brain_tumor_dataset.csv')
df.info()
# target column == Tumor_type (Malignant or Benign)

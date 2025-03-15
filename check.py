import pandas as pd
import numpy as np
from utils.utils import read_cfg


cfg = read_cfg(cfg_file='D:\\skin\\config\\config.yaml')

csv_file = cfg['train_set']

df = pd.read_csv(csv_file)

print(df.value_counts(df['dx']))
import pandas as pd

df = pd.read_csv('D:/YOLO_Custom/runs/detect/flir_4class_v13/results.csv')
df.columns = df.columns.str.strip()
print(df.columns.tolist())
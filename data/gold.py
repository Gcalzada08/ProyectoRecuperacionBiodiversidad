import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Cargar datos limpios desde silver
df = pd.read_csv("data/silver/clima_clean.csv")

# Transformaciones cíclicas de fechas
df['hora_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['fecha']).dt.hour / 24)
df['hora_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['fecha']).dt.hour / 24)

# Escalado robusto
scaler = RobustScaler()
scaled = scaler.fit_transform(df[['temperatura','precipitacion','humedad']])
df_scaled = pd.DataFrame(scaled, columns=['temp_scaled','prec_scaled','hum_scaled'])

# Dataset final
df_final = pd.concat([df, df_scaled], axis=1)
df_final.to_csv("data/gold/dataset_final.csv", index=False)

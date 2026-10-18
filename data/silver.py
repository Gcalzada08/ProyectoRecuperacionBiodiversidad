import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

# Cargar datos crudos desde bronze
df = pd.read_csv("data/bronze/clima_raw.csv")

# Imputación de nulos con KNN
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Detección de outliers con Isolation Forest
iso = IsolationForest(contamination=0.02, random_state=42)
df['outlier'] = iso.fit_predict(df_imputed)

# Guardar datos limpios en silver
df.to_csv("data/silver/clima_clean.csv", index=False)

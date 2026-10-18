# Figura 2.2. Código de ingesta de las tres fuentes

import pandas as pd
import json
from pathlib import Path

# Este script está en notebooks; la carpeta data está en la raíz del proyecto.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# -------------------------------
# 1. Ingesta de datos estructurados (SQL / CSV)
# -------------------------------
# Ejemplo: datos climáticos (temperatura, precipitación, humedad)
clima = pd.read_csv(DATA_DIR / "bronze" / "clima_raw.csv")
print("Clima (primeras filas):")
print(clima.head())

# -------------------------------
# 2. Ingesta de datos semi-estructurados (API REST / JSON)
# -------------------------------
# Ejemplo: registros de biodiversidad desde GBIF o API meteorológica
with open(DATA_DIR / "bronze" / "biodiversidad_raw.json", "r", encoding="utf-8") as f:
    biodiversidad = json.load(f)

print("\nBiodiversidad (ejemplo de registro):")
print(biodiversidad)

# -------------------------------
# 3. Ingesta de datos no estructurados (IoT / logs de sensores)
# -------------------------------
# Ejemplo: sensores acústicos o ambientales
with open(DATA_DIR / "bronze" / "sensores_raw.log", "r", encoding="utf-8") as f:
    sensores = f.readlines()

print("\nSensores (primeras 5 líneas):")
print(sensores[:5])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos limpios desde Silver; si aún no existen, usar el bronze como fallback
silver_path = DATA_DIR / "silver" / "clima_clean.csv"
bronze_path = DATA_DIR / "bronze" / "clima_raw.csv"

if silver_path.exists():
    df = pd.read_csv(silver_path)
else:
    df = pd.read_csv(bronze_path)

df = df.copy()

if "fecha" in df.columns:
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

# 1. Calcular % de nulos por variable
missing_data = df.isnull().mean() * 100
missing_table = pd.DataFrame({
    "Variable": missing_data.index,
    "% de nulos": missing_data.values
})
print(missing_table)

# 2. Visualizar nulos con mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Mapa de calor de valores faltantes")
plt.show()

from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

# 1. Imputación de valores faltantes con KNN
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if not numeric_cols:
    raise ValueError("No se encontraron columnas numéricas para imputar.")

imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

# 2. Detección de outliers con Isolation Forest
iso = IsolationForest(contamination=0.02, random_state=42)
df["outlier"] = iso.fit_predict(df[numeric_cols])

# Guardar datos limpios en Silver
silver_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(silver_path, index=False)

import numpy as np
from sklearn.preprocessing import RobustScaler

# 1. Transformaciones cíclicas de fecha
if "fecha" in df.columns:
    df["hora_sin"] = np.sin(2 * np.pi * df["fecha"].dt.hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["fecha"].dt.hour / 24)
    df["dia_sin"] = np.sin(2 * np.pi * df["fecha"].dt.dayofyear / 365.25)
    df["dia_cos"] = np.cos(2 * np.pi * df["fecha"].dt.dayofyear / 365.25)

# 2. Escalado robusto de variables climáticas
climate_cols = [c for c in ["temperatura", "precipitacion", "humedad"] if c in df.columns]
if not climate_cols:
    climate_cols = [c for c in numeric_cols if c != "outlier"]

scaler = RobustScaler()
scaled = scaler.fit_transform(df[climate_cols])
df_scaled = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in climate_cols])

# 3. Dataset final en Gold
out_dir = DATA_DIR / "gold"
out_dir.mkdir(parents=True, exist_ok=True)
df_final = pd.concat([df, df_scaled], axis=1)
df_final.to_csv(out_dir / "dataset_final.csv", index=False)

import pandas as pd

df = pd.read_csv("data/bronze/clima_raw.csv")

# Mostrar todas las filas
print(df.to_string())

import matplotlib.pyplot as plt

# Comparar distribución antes y después
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
df['humedad_media'].hist(bins=10, color='skyblue')
plt.title("Antes de imputación/outliers")

plt.subplot(1,2,2)
pd.DataFrame(
    RobustScaler().fit_transform(df[['temp_min', 'temp_max', 'humedad_media', 'precipitacion_mm']]),
    columns=['temp_min', 'temp_max', 'humedad_media', 'precipitacion_mm']
)['humedad_media'].hist(bins=10, color='salmon')
plt.title("Después de imputación/outliers")

plt.show()

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# 1. Cargar datos desde Bronze
df = pd.read_csv("data/bronze/clima_raw.csv")

# 2. Seleccionar variables numéricas para imputación
X = df[['temp_min','temp_max','humedad_media','precipitacion_mm']]

# 3. Imputación de valores faltantes con KNN (k=5)
imputer = KNNImputer(n_neighbors=5)
X_imp = imputer.fit_transform(X)

# 4. Detección de outliers con Isolation Forest
iso = IsolationForest(contamination=0.02, random_state=42)
etiquetas = iso.fit_predict(X_imp)   # -1 = atípico, 1 = normal
df['outlier'] = etiquetas

# 5. Escalado robusto de variables
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imp)

# 6. Crear DataFrame final con variables escaladas
df_scaled = pd.DataFrame(X_scaled, columns=['temp_min_scaled','temp_max_scaled','humedad_scaled','precip_scaled'])
df_final = pd.concat([df, df_scaled], axis=1)

# 7. Guardar en Silver
df_final.to_csv("data/silver/clima_clean.csv", index=False)

print(" Limpieza e imputación completadas. Archivo guardado en data/silver/clima_clean.csv")
print(df_final.head())

# Mostrar todas las filas
print(df_final.to_string())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(df[['temp_min','temp_max','humedad_media','precipitacion_mm']].corr(),
            annot=True, cmap='coolwarm', center=0)
plt.title("Mapa de calor de correlación")
plt.show()

df[['temp_min','temp_max','humedad_media','precipitacion_mm']].hist(bins=10, figsize=(10,6), color='skyblue')
plt.suptitle("Histogramas de distribución de variables de hábitat")
plt.show()

sns.pairplot(df[['temp_min','temp_max','humedad_media','precipitacion_mm']], diag_kind='kde')
plt.suptitle("Diagramas de dispersión multivariantes", y=1.02)
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Variables seleccionadas
X = df[['temp_min','temp_max','humedad_media','precipitacion_mm']]

# Escalado robusto
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Modelo K-Means
kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++', max_iter=300)
labels = kmeans.fit_predict(X_scaled)

# Métrica de silueta
sil_score = silhouette_score(X_scaled, labels)
print("Coeficiente de silueta:", sil_score)

# Visualización de clusters
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis')
plt.title("Clusters K-Means (temp_min vs temp_max)")
plt.xlabel("temp_min (scaled)")
plt.ylabel("temp_max (scaled)")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Supongamos que tu DataFrame tiene columnas: latitud, longitud, zona
# zona puede ser 'urbana', 'turística', 'selvática', etc.
# Ejemplo:
# df = pd.DataFrame({
#     'lat': [19.3, 19.4, 18.9, 17.5],
#     'lon': [-99.1, -99.2, -98.7, -96.8],
#     'zona': ['urbana','urbana','turística','selvática']
# })

geo_columns = {"lon", "lat", "zona"}
if geo_columns.issubset(df.columns):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='lon', y='lat', hue='zona', palette='Set1', s=80)

    plt.title("Figura 5.1. Evidencia del sesgo: distribución geográfica de los registros")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend(title="Zona")
    plt.show()
else:
    missing_columns = ", ".join(sorted(geo_columns - set(df.columns)))
    print(
        "\nNo se generó la visualización geográfica: "
        f"faltan las columnas {missing_columns} en los datos."
    )

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Crear un DataFrame de ejemplo con zonas y coordenadas simuladas
np.random.seed(42)

n_urbana = 50
n_turistica = 30
n_selvatica = 10

df = pd.DataFrame({
    'lat': np.concatenate([
        np.random.normal(19.3, 0.05, n_urbana),     # zona urbana (ej. CDMX)
        np.random.normal(20.5, 0.05, n_turistica),  # zona turística (ej. Cancún)
        np.random.normal(17.0, 0.05, n_selvatica)   # zona selvática (ej. Chiapas)
    ]),
    'lon': np.concatenate([
        np.random.normal(-99.1, 0.05, n_urbana),
        np.random.normal(-86.8, 0.05, n_turistica),
        np.random.normal(-92.6, 0.05, n_selvatica)
    ]),
    'zona': (['urbana']*n_urbana + ['turística']*n_turistica + ['selvática']*n_selvatica)
})

# Graficar distribución geográfica
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='lon', y='lat', hue='zona', palette='Set1', s=80)

plt.title("Figura 5.1. Evidencia del sesgo: distribución geográfica de los registros")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.legend(title="Zona")
plt.show()

import importlib.util
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Preparar los datos geográficos y dividirlos en entrenamiento y prueba
X = df[["lat", "lon"]]
y = df["zona"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Explicabilidad con SHAP (opcional)
if importlib.util.find_spec("shap") is not None:
    shap = importlib.import_module("shap")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    class_index = 1
    if isinstance(shap_values, list):
        class_shap_values = shap_values[class_index]
    elif shap_values.ndim == 3:
        class_shap_values = shap_values[:, :, class_index]
    else:
        class_shap_values = shap_values

    shap.summary_plot(class_shap_values, X_test, plot_type="bar")
else:
    print("SHAP no está instalado; se omite la gráfica de explicabilidad.")

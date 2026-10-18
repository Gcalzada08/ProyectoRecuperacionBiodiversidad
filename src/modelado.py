import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Cargar dataset final desde gold
df = pd.read_csv("data/gold/dataset_final.csv")

# 2. Definir variables de entrada (X) y variable objetivo (y)
X = df[['temp_scaled', 'prec_scaled', 'hum_scaled', 'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos']]
y = df['presencia_polinizadores']  # 1 = presencia, 0 = ausencia

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Entrenar modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=200,       # número de árboles
    max_depth=10,           # profundidad máxima
    random_state=42,
    class_weight="balanced" # balancear clases desiguales
)
rf.fit(X_train, y_train)

# 5. Predicciones
y_pred = rf.predict(X_test)

# 6. Evaluación del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("\nReporte completo:\n", classification_report(y_test, y_pred))

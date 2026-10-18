import pandas as pd
import json

# cargar datos climáticos crudos
clima_raw = pd.read_csv("data/bronze/clima_raw.csv")

# cargar datos de biodiversidad crudos
with open("data/bronze/biodiversidad_raw.json", "r") as f:
    biodiversidad_raw = json.load(f)

# Guardar respaldo en bronze
clima_raw.to_csv("data/bronze/clima_raw_backup.csv", index=False)

import os
print(os.getcwd())
print(os.path.exists("data/bronze/clima_raw.csv"))
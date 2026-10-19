# ProyectoRecuperacionBiodiversidad

# Proyecto de Biodiversidad - Polinizadores en Xochimilco

## Estructura
- data/bronze: datos crudos
- data/silver: datos limpios e integrados
- data/gold: dataset final listo para modelar
- notebooks/: análisis exploratorio y modelado
- src/: scripts de procesamiento
- README.md: documentación del proyecto

## Ejecutar la aplicación

Instala las dependencias con `python -m pip install -r requirements.txt` y ejecuta
`python -m streamlit run notebooks/proyecto_biodiversidad.py`. Streamlit Community
Cloud instala automáticamente las dependencias declaradas en `requirements.txt`
ubicado en la raíz del repositorio.

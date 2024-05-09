import pdfplumber
import pandas as pd
import re

# Define el camino al archivo PDF
pdf_path = 'Ejemplo.pdf'

# Lista para almacenar cada entrada de datos
data = []

# Expresión regular para identificar frases que podrían ser TDR
# Este es un ejemplo genérico; deberás ajustarlo según los patrones específicos de tu documento
tdr_pattern = re.compile(r'\b(definición|análisis|diseño|desarrollo|integración|implementación|administración|seguridad)\b', re.IGNORECASE)

# Abrir el archivo PDF con pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    # Iterar sobre cada página del PDF
    for page in pdf.pages:
        # Extraer texto de cada página
        text = page.extract_text()
        # Buscar todas las ocurrencias del patrón TDR en el texto
        for match in tdr_pattern.finditer(text):
            start, end = match.span()
            # Intentar capturar la oración completa que contiene el TDR
            sentence = text[:end].split('. ')[-1] + text[end:].split('. ')[0]
            data.append({"TDR": sentence.strip()})

# Convertir la lista de datos a DataFrame
df = pd.DataFrame(data)

# Agregar columna de ID
df['Código'] = ['TDR-' + str(i+1) for i in range(len(df))]

# Guardar el DataFrame en un archivo Excel
df.to_excel('salida.xlsx', index=False)
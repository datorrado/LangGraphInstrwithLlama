import pandas as pd

import os
print("Carpeta de trabajo:", os.getcwd())
# Lee el archivo original (delimitado por coma)
df = pd.read_csv('c:\\Users\\david\\OneDrive\\Documentos\\GitHub\\LangGraphInstrwithLlama\\3Hour\\20_5\\ActiveThreadsOT2.csv')

# Mantén solo las columnas 'timeStamp' y 'allThreads'
df_simple = df[['timeStamp', 'allThreads']]

# Guarda el resultado en un nuevo archivo CSV
df_simple.to_csv('ActiveThreads_simple.csv', index=False)

print('Archivo guardado como ActiveThreads_simple.csv')


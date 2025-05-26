import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV
df = pd.read_csv('EvalResults/10Node/ResultsTable.csv')

# Convertir timestamps a segundos desde el inicio
df['arrival_time'] = (df['timeStamp'] - df['timeStamp'].min()) / 1000.0

# Ordenar por tiempo (por si acaso)
df = df.sort_values('arrival_time').reset_index(drop=True)

# Calcular diferencias de tiempo (inter-arrival)
df['delta'] = df['arrival_time'].diff()

# Crear figura
plt.figure(figsize=(16, 4))
plt.vlines(df['arrival_time'], ymin=0, ymax=1, color='blue', linewidth=8)

# Dibujar líneas horizontales entre cada par de arrivals
for i in range(1, len(df)):
    x0 = df['arrival_time'][i-1]
    x1 = df['arrival_time'][i]
    y = 0.50  # Un poco por encima de las barras verticales

    # Línea horizontal entre arrivals
    plt.hlines(y, x0, x1, color='gray', linestyle='--')

    # Etiqueta con el delta
    delta = round(x1 - x0, 2)
    plt.text((x0 + x1) / 2, y + 0.02, f't{i}:{delta}s', ha='center', fontsize=6, rotation=45)


# Decoración
plt.xlabel('Tiempo (s)')
plt.title('Arrival Timeline con Intervalos entre Requests')
plt.yticks([])  # Oculta eje Y
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()
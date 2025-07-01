import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import timedelta

import matplotlib.dates as mdates



folder1 = Path("3Hour") / "15_5" / "ResponseTime.csv"
folder2 = Path("3Hour") / "5_5" / "tool_evaluations_5.csv"

folder = "energy_summary.csv"
def calcular_promedio_response_time(filepath, nodos, usuarios, output_path="response_summary.csv"):
    """
    Calcula el promedio de respuesta desde un archivo CSV JMeter y lo guarda con nodos y usuarios como referencia.
    """
    df = pd.read_csv(filepath, sep=';')
    df.columns = ['Elapsed time', 'my_test_sampler']
    df['my_test_sampler'] = pd.to_numeric(df['my_test_sampler'], errors='coerce')

    promedio = df['my_test_sampler'].mean()
    promedio = promedio/1000  # Convertir a segundos
    nueva_fila = pd.DataFrame({
        'nodes': [nodos],
        'users': [usuarios],
        'avg': [promedio]
    })

    if os.path.exists(output_path):
        existente = pd.read_csv(output_path)
        resultado = pd.concat([existente, nueva_fila], ignore_index=True)
    else:
        resultado = nueva_fila

    resultado.to_csv(output_path, index=False)

def convertir_a_segundos(hms):
    h, m, s = hms.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def formatear_hhmmss(seg):
    horas = int(seg // 3600)
    minutos = int((seg % 3600) // 60)
    segundos = int(seg % 60)
    return f"{horas:02}:{minutos:02}:{segundos:02}"

def procesar_y_graficar_utilization(filepath, usuarios, nodos):
    df = pd.read_csv(filepath, sep=';')
    df.columns = df.columns.str.strip()
    df['Elapsed time'] = df['Elapsed time'].astype(str).str.strip()

    # Convertir Elapsed time a segundos
    df['segundos'] = df['Elapsed time'].apply(convertir_a_segundos)

    # Corregir reinicio de día (cuando pasa de 23:59:59 a 00:00:00)
    segundos = df['segundos'].tolist()
    segundos_corr = []
    acumulado = 0

    for i in range(len(segundos)):
        if i > 0 and segundos[i] < segundos[i - 1]:
            acumulado += 86400  # se reinició el reloj (nuevo día)
        segundos_corr.append(segundos[i] + acumulado)

    # Calcular tiempo transcurrido desde el inicio
    tiempo_transcurrido = [max(0, s - segundos_corr[0]) for s in segundos_corr]

    df['Tiempo transcurrido'] = tiempo_transcurrido

    # Limpiar columnas
    for col in ['CPU', 'GPU1', 'GPU2']:
        df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Tiempo transcurrido'])

    if df.empty:
        print("❌ No hay datos válidos.")
        return

    # Agrupar cada 10 filas (cada 10 segundos aprox)
    df_grouped = df.groupby(df.index // 10).agg({
        'Tiempo transcurrido': 'first',
        'CPU': 'mean',
        'GPU1': 'mean',
        'GPU2': 'mean'
    }).reset_index(drop=True)

    # Convertir segundos a formato HH:MM:SS para etiquetas
    df_grouped['Tiempo'] = df_grouped['Tiempo transcurrido'].apply(formatear_hhmmss)
    df_grouped = df_grouped.rename(columns={'GPU1': 'GPU 1', 'GPU2': 'GPU 2'})

    # Derretir para graficar
    df_melted = df_grouped.melt(id_vars='Tiempo', value_vars=['CPU', 'GPU 1', 'GPU 2'],
                                var_name='Componente', value_name='Uso (%)')

    # Graficar
    plt.figure(figsize=(16, 6))
    ax = sns.lineplot(data=df_melted, x='Tiempo', y='Uso (%)', hue='Componente')

    # Mostrar etiquetas cada 30 minutos
    total = len(df_grouped)
    intervalo = 180 if total > 180 else max(1, total // 15)
    visibles = df_grouped['Tiempo'].iloc[::intervalo].tolist()

    for label in ax.get_xticklabels():
        if label.get_text() not in visibles:
            label.set_visible(False)

    plt.title("Utilization for components over time, for {} users and {} nodes".format(usuarios, nodos))
    plt.xlabel("Elapsed Time (HH:MM:SS)")
    plt.ylabel("Utilization Percentage")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_energy_by_tool(df):
    metrics = ["CPU", "GPU", "RAM", "Total"]
    tools = df["tool_name"].unique()
    
    plt.figure(figsize=(18, 12))
    
    for i, metric in enumerate(metrics, 1):
        ax = plt.subplot(2, 2, i)
        metric_df = df[df["Metric"] == metric]
        
        for tool in tools:
            tool_df = metric_df[metric_df["tool_name"] == tool].sort_values("users")
            ax.plot(tool_df["users"], tool_df["mean_value"], marker="o", label=tool)
            
            # Annotate each point with mean_value
            for _, row in tool_df.iterrows():
                ax.annotate(f'{row["mean_value"]:.2e}', 
                            (row["users"], row["mean_value"]), 
                            textcoords="offset points", 
                            xytext=(0,5), 
                            ha='center', 
                            fontsize=8, 
                            fontweight='bold')

        ax.set_title(f"{metric} Energy Consumption")
        ax.set_xlabel("Number of Users")
        ax.set_ylabel("Mean Energy Value (kWh)")
        ax.grid(True)
        
        # Only show legend in the first plot
        if i == 1:
            ax.legend()
    
    plt.tight_layout()
    plt.show()
folder3 = Path("3Hour") / "1_5" / "Utilization.csv"
df = pd.read_csv(folder, sep=',')
paths = ["10_3", "10_5", "20_3", "20_5", "30_3", "30_5", "40_3", "40_5", "50_3", "50_5"]
for path in paths:
    path = str(path)
    antes, despues = path.split('_')
    antes = int(antes)
    despues = int(despues)
    folder = Path("3Hour") / path / "Utilization.csv"   
    procesar_y_graficar_utilization(folder, antes, despues)
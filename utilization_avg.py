import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

import os
import glob

# Construir la ruta usando os.path.join
base_dir = os.path.join('EvalResults', '10Node')
pattern = os.path.join(base_dir, 'Utilization*.csv')

# Buscar archivos
file_paths = glob.glob(pattern)

# Confirmar que encontró archivos
for path in file_paths:
    print(f"\nArchivo: {path}")
    try:
        df = pd.read_csv(path, delimiter=';', engine='python')
        print("Columnas:", df.columns)
        print(df.head(5))
    except Exception as e:
        print(f"Error al leer {path}: {e}")

# Paso 1: obtener el menor tiempo de todos los archivos
def get_min_time(path):
    df = pd.read_csv(path, delimiter=';', engine='python')
    df['Elapsed time'] = df['Elapsed time'].str.replace(',', '.', regex=False)
    df['Elapsed time'] = pd.to_datetime(df['Elapsed time'], format="%H:%M:%S.%f", errors='coerce')
    df = df.dropna(subset=['Elapsed time'])
    return df['Elapsed time'].min() if not df.empty else None

# Obtener tiempo global mínimo
all_times = [get_min_time(p) for p in file_paths]
global_start_time = min([t for t in all_times if t is not None])

# Paso 2: función para procesar con tiempo global
def safe_process_df(path):
    try:
        df = pd.read_csv(path, delimiter=';', engine='python')
        df.columns = [col.strip() for col in df.columns]
        df['Elapsed time'] = df['Elapsed time'].str.replace(',', '.', regex=False)
        df['Elapsed time'] = pd.to_datetime(df['Elapsed time'], format="%H:%M:%S.%f", errors='coerce')
        df = df.dropna(subset=['Elapsed time'])
        if df.empty:
            return pd.DataFrame()

        # Calcular minutos desde tiempo mínimo global
        df['Minutes'] = ((df['Elapsed time'] - global_start_time).dt.total_seconds() // 60).astype(int)

        for col in ['CPU', 'GPU1', 'GPU2']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.groupby('Minutes')[['CPU', 'GPU1', 'GPU2']].mean()
    except Exception as e:
        print(f"Error en {path}: {e}")
        return pd.DataFrame()



# Procesar todos los archivos
processed_dfs = [safe_process_df(path) for path in file_paths if safe_process_df(path).shape[0] > 0]

# Unir y graficar si hay datos válidos
if processed_dfs:
    combined_df = pd.concat(processed_dfs).groupby(level=0).mean()

    # Preparar para Seaborn
    plot_df = combined_df.reset_index().melt(id_vars='Minutes', 
                                             value_vars=['CPU', 'GPU1', 'GPU2'],
                                             var_name='Componente',
                                             value_name='Uso (%)')

    # Gráfica
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=plot_df, x='Minutes', y='Uso (%)', hue='Componente')
    plt.title('Promedio de Uso de CPU y GPUs por Minuto')
    plt.xlabel('Minutos desde inicio')
    plt.ylabel('Uso promedio (%)')
    plt.legend(title='Componente')
    plt.tight_layout()
    plt.show()
else:
    print("No se encontraron datos válidos en los archivos.")


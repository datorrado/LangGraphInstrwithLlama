# Imagen base optimizada para Python y notebooks
FROM python:3.11-slim

# Evita prompts interactivos (útil para instalar)
ENV DEBIAN_FRONTEND=noninteractive

# Instalar paquetes del sistema necesarios
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . .

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter y notebooks
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

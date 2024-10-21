import os
import pdfplumber
import re
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

app = Flask(__name__)

# Almacenamiento global para los datos de los nadadores
datos_nadadores = []
ruta_pdf = "./uploads/"
ruta_png = "static/"

# Función para convertir segundos a formato MM:SS
def segundos_a_minutos_segundos(segundos):
    minutos = int(segundos // 60)
    segundos_restantes = segundos % 60
    return f"{minutos}:{segundos_restantes:05.2f}"  # Formato MM:SS

# Función para convertir tiempo a segundos
def tiempo_a_segundos(tiempo):
    tiempo = tiempo.replace(',', '.')  # Cambiar coma por punto
    if ':' not in tiempo:
        return float(tiempo)

    partes = tiempo.split(':')
    if len(partes) == 2:  # Formato MM:SS
        minutos = int(partes[0])
        segundos = float(partes[1])
        return minutos * 60 + segundos
    elif len(partes) == 3:  # Formato HH:MM:SS
        horas = int(partes[0])
        minutos = int(partes[1])
        segundos = float(partes[2])
        return horas * 3600 + minutos * 60 + segundos
    else:
        return float(tiempo)

# Función para convertir string a fecha
def convertir_a_fecha(fecha_str):
    return datetime.strptime(fecha_str, "%d/%m/%Y")

# Función para extraer datos del PDF
def extraer_datos_pdf(pdf_path):
    nadador_info = {}
    resultados = []

    with pdfplumber.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text()
            lineas = texto.split("\n")

            estilo = None
            distancia = 0

            for i, line in enumerate(lineas):
                if any(style in line for style in ['Free', 'Back', 'Breast', 'Fly', 'IM']):
                    partes = line.split()
                    if len(partes) >= 2:
                        distancia = int(partes[0])
                        print(distancia+1)
                        estilo = partes[1]

                if estilo and distancia:
                    while i + 1 < len(lineas):
                        siguiente_linea = lineas[i + 1].split()
                        if len(siguiente_linea) >= 3:
                            tiempo_completo = siguiente_linea[0]
                            tiempo = re.match(r'[\d:,]+', tiempo_completo).group()
                            tiempo_segundos = tiempo_a_segundos(tiempo)

                            fecha_str = siguiente_linea[2]
                            if fecha_str == "F":
                                fecha_str = siguiente_linea[3]
                                i += 1

                            fecha = convertir_a_fecha(fecha_str)
                            competencia = " ".join(siguiente_linea[3:]) if fecha_str != "F" else " ".join(siguiente_linea[4:])

                            resultado = {
                                'Estilo': estilo,
                                'Distancia': distancia,
                                'Tiempo': tiempo_segundos,
                                'Fecha': fecha,
                                'Competencia': competencia
                            }

                            if resultado not in resultados:
                                resultados.append(resultado)

                            i += 1
                        else:
                            break

                    estilo = None
                    distancia = None

            if not nadador_info.get('Nombre'):
                nombre_line = [line for line in lineas if "Top Times Report for" in line]
                if nombre_line:
                    nadador_info['Nombre'] = nombre_line[0].split("for")[-1].strip()

    nadador_info['Resultados'] = resultados
    return nadador_info

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar la subida de archivos
@app.route('/upload')
def upload_file():
    # Limpiar datos anteriores
    global datos_nadadores
    datos_nadadores = []

    # Obtener todos los archivos PDF en la carpeta 'uploads'
    archivos_pdf = [f for f in os.listdir(ruta_pdf) if f.endswith('.pdf')]

    for archivo in archivos_pdf:
        file_path = os.path.join(ruta_pdf, archivo)  # Ruta del archivo PDF
        datos_nadadores.append(extraer_datos_pdf(file_path))

    # Ordenar los nadadores por nombre antes de enviarlos al render
    datos_nadadores = sorted(datos_nadadores, key=lambda x: x['Nombre'])

    return render_template('select_nadador.html', nadadores=datos_nadadores)


# Ruta para mostrar gráficos del nadador seleccionado
@app.route('/graficos', methods=['GET'])
def graficos():
    nombre_nadador = request.args.get('nombre_nadador')
    tipo_grafico = request.args.get('tipo_grafico')
    nadador = next((n for n in datos_nadadores if n['Nombre'] == nombre_nadador), None)

    if nadador:
        graficos_paths = []
        if tipo_grafico == 'estilo_tiempo_fecha':
            graficos_paths = generar_graficos_individuales([nadador])
        elif tipo_grafico == 'rendimiento':
            graficos_paths = generar_grafico_rendimiento(nadador)
        elif tipo_grafico == 'promedio_de_pases_en_100m':
            graficos_paths.append(generar_grafico_tiempo_promedio_por_100_metros(nadador))
        elif tipo_grafico == 'minimo_de_pases_en_100m':
            graficos_paths.append(generar_grafico_tiempo_minimo_por_100_metros(nadador))



        return render_template('select_nadador.html', nadadores=datos_nadadores, graficos=graficos_paths, nombre_nadador=nombre_nadador)
    return "Nadador no encontrado", 404


def segundos_a_minutos_segundos(segundos):
    """Convierte segundos a formato MM:SS."""
    minutos = int(segundos // 60)
    seg = int(segundos % 60)
    return f'{minutos:02}:{seg:02}'

# Función para generar gráficos de área individuales con puntos
def generar_graficos_individuales(datos):
    graficos_paths = []
    for nadador in datos:
        df = pd.DataFrame(nadador['Resultados'])
        df.sort_values(by='Fecha', inplace=True)

        for estilo in df['Estilo'].unique():
            for distancia in df['Distancia'].unique():
                subset = df[(df['Estilo'] == estilo) & (df['Distancia'] == distancia)]
                if not subset.empty:
                    plt.figure(figsize=(10, 6))
                    plt.fill_between(subset['Fecha'], subset['Tiempo'], alpha=0.5)
                    plt.plot(subset['Fecha'], subset['Tiempo'], 'o', color='red')
                    plt.title(f'Tiempos de {nadador["Nombre"]} - Estilo: {estilo}, Distancia: {distancia}')
                    plt.xlabel('Fecha')
                    plt.ylabel('Tiempo (MM:SS)')
                    plt.xticks(rotation=45)

                    # Ajustar el rango del eje Y
                    min_tiempo = subset['Tiempo'].min()
                    max_tiempo = subset['Tiempo'].max()
                    plt.ylim(min_tiempo * 0.95, max_tiempo * 1.01)  # Establecer límites un poco antes del mínimo y un poco más del máximo

                    # Colocar etiquetas en el eje Y en formato MM:SS
                    y_ticks = np.linspace(min_tiempo, max_tiempo, 4)  # Generar 4 ticks en el eje Y
                    plt.yticks(y_ticks, labels=[segundos_a_minutos_segundos(t) for t in y_ticks])

                    plt.tight_layout()
                    grafico_path = f'{ruta_png}{nadador["Nombre"]}_{estilo}_{distancia}.png'
                    plt.savefig(grafico_path)
                    plt.close()
                    graficos_paths.append(grafico_path)
    return graficos_paths

# Función para generar gráfico de rendimiento (velocidad promedio)
def generar_grafico_rendimiento(nadador):
    df = pd.DataFrame(nadador['Resultados'])
    
    # Calcular la velocidad (distancia / tiempo)
    df['Velocidad'] = df['Distancia'] / df['Tiempo']  # Velocidad en metros por segundo

    estilos = df['Estilo'].unique()
    velocidades = [df[df['Estilo'] == estilo]['Velocidad'].mean() for estilo in estilos]

    # Crear gráfico de radar
    num_vars = len(estilos)

    # Crear un ángulo para cada variable
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Completar el círculo
    velocidades += velocidades[:1]
    angles += angles[:1]

    plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, velocidades, color='blue', alpha=0.25)
    ax.plot(angles, velocidades, color='blue', linewidth=2)

    # Etiquetas
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(estilos)

    plt.title(f'Rendimiento de {nadador["Nombre"]} (Velocidad Promedio)', size=15)
    grafico_path = f'{ruta_png}{nadador["Nombre"]}_rendimiento_velocidad.png'
    plt.savefig(grafico_path)
    plt.close()
    return [grafico_path]


# Función para generar gráfico de pases minimos en 100m
def generar_grafico_tiempo_minimo_por_100_metros(nadador):
    df = pd.DataFrame(nadador['Resultados'])
    
    # Calcular el tiempo promedio por cada 100 metros
    df['Tiempo_por_100'] = df['Tiempo'] / (df['Distancia'] / 100)
    
    # Agrupar por Estilo y Distancia
    rendimiento = df.groupby(['Estilo', 'Distancia']).agg({'Tiempo_por_100': 'min'}).reset_index()
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    for estilo in rendimiento['Estilo'].unique():
        subset = rendimiento[rendimiento['Estilo'] == estilo]
        plt.plot(subset['Distancia'], subset['Tiempo_por_100'], marker='o', label=f'{estilo}')
    
    plt.title(f'Tiempo Promedio por 100 Metros de {nadador["Nombre"]}')
    plt.xlabel('Distancia (metros)')
    plt.ylabel('Tiempo Mínimo por 100 Metros (MM:SS)')
    plt.xticks(rotation=45)
    plt.legend()

    # Cambiar los ticks del eje Y a formato MM:SS
    plt.yticks(ticks=plt.yticks()[0], labels=[segundos_a_minutos_segundos(t) for t in plt.yticks()[0]])

    plt.tight_layout()
    
    grafico_path = f'{ruta_png}{nadador["Nombre"]}_tiempo_promedio_por_100_metros.png'
    plt.savefig(grafico_path)
    plt.close()
    return grafico_path

# Función para generar gráfico de pases promedio en 100m
def generar_grafico_tiempo_promedio_por_100_metros(nadador):
    df = pd.DataFrame(nadador['Resultados'])
    
    # Calcular el tiempo promedio por cada 100 metros
    df['Tiempo_por_100'] = df['Tiempo'] / (df['Distancia'] / 100)
    
    # Agrupar por Estilo y Distancia
    rendimiento = df.groupby(['Estilo', 'Distancia']).agg({'Tiempo_por_100': 'mean'}).reset_index()
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    for estilo in rendimiento['Estilo'].unique():
        subset = rendimiento[rendimiento['Estilo'] == estilo]
        plt.plot(subset['Distancia'], subset['Tiempo_por_100'], marker='o', label=f'{estilo}')
    
    plt.title(f'Tiempo Promedio por 100 Metros de {nadador["Nombre"]}')
    plt.xlabel('Distancia (metros)')
    plt.ylabel('Tiempo Promedio por 100 Metros (MM:SS)')
    plt.xticks(rotation=45)
    plt.legend()

    # Cambiar los ticks del eje Y a formato MM:SS
    plt.yticks(ticks=plt.yticks()[0], labels=[segundos_a_minutos_segundos(t) for t in plt.yticks()[0]])

    plt.tight_layout()
    
    grafico_path = f'{ruta_png}{nadador["Nombre"]}_tiempo_promedio_por_100_metros.png'
    plt.savefig(grafico_path)
    plt.close()
    return grafico_path



if __name__ == '__main__':
    app.run(debug=True)

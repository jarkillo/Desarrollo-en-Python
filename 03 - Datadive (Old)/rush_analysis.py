import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations


def rush_analysis(file_path):
    # Cargar el archivo Excel
    data = pd.read_excel(file_path)

    # Crear una carpeta para guardar los gráficos si no existe
    folder_name = 'rush_analysis_output'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Obtener todas las combinaciones de columnas numéricas para los gráficos
    numeric_cols = data.select_dtypes(include='number').columns
    col_pairs = list(combinations(numeric_cols, 2))

    # Función para verificar si un archivo ya existe
    def save_plot_if_not_exists(fig, file_name):
        if not os.path.isfile(file_name):
            fig.savefig(file_name)
        else:
            print(f"Archivo '{file_name}' ya existe. No se guarda.")

    # Generar gráficos para cada par de columnas
    for pair in col_pairs:
        x, y = pair

        # Gráfico de dispersión
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=x, y=y)
        plt.title(f'Scatter Plot - {x} vs {y}')
        scatter_file = f'{folder_name}/scatter_{x}_vs_{y}.png'
        save_plot_if_not_exists(plt, scatter_file)
        plt.close()

        # Gráfico de densidad (solo si ambas columnas tienen suficiente variabilidad)
        if data[x].nunique() > 1 and data[y].nunique() > 1:
            try:
                sns.jointplot(data=data, x=x, y=y, kind="kde")
                plt.title(f'Density Plot - {x} vs {y}', pad=70)
                density_file = f'{folder_name}/density_{x}_vs_{y}.png'
                save_plot_if_not_exists(plt, density_file)
            except ValueError:
                print(
                    f"No se pudo generar el gráfico de densidad para {x} vs {y} debido a problemas de nivel de contorno.")
            plt.close()

    # Mapa de calor para la correlación
    plt.figure(figsize=(12, 10))
    corr = data[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Correlation')
    heatmap_file = f'{folder_name}/heatmap_correlation.png'
    save_plot_if_not_exists(plt, heatmap_file)
    plt.close()

    print("Análisis completado. Gráficos guardados o ya existentes.")


rush_analysis("Diamonds_sin_atipicos.xlsx")

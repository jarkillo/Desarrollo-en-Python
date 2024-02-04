from data_manager import load_data, save_data  # Asegúrate de que esto utiliza la versión corregida de load_data
from classes import RaidCalculator  # Asume que tienes una clase RaidCalculator

def update_damage_per_explosive(filename='raid_calculator_data.json'):
    # Crear una instancia del calculador de asaltos
    raid_calculator = RaidCalculator()

    # Cargar datos existentes
    load_data(raid_calculator, filename)

    # Recalcular damage_per_explosive para cada estructura
    for structure in raid_calculator.structures:
        structure.calculate_damage_per_structure()

    # Guardar los datos actualizados
    save_data(raid_calculator, filename)  # Ajusta para usar el método save_data

# Ejecutar la función de actualización
update_damage_per_explosive()

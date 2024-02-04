import json
import os
from classes import Structure, Explosive, RaidCalculator

def save_data(raid_calculator: RaidCalculator, filename='raid_calculator_data.json'):
    # Convertir el set a list para serializar
    config_for_json = raid_calculator.config.copy()
    config_for_json['exclude_explosives'] = list(config_for_json['exclude_explosives'])

    # Estructura de los datos a ser guardados
    data = {
        "structures": [
            structure.to_dict() for structure in raid_calculator.structures
        ],
        "explosives": [
            explosive.to_dict() for explosive in raid_calculator.explosives
        ],
        "config": config_for_json  # Usa la configuración convertida para JSON
    }
    
    # Guardar los datos en un archivo JSON
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def load_data(raid_calculator: RaidCalculator, filename='raid_calculator_data.json'):
    try:
        # Abre el archivo JSON para cargar los datos
        with open(filename, 'r') as file:
            data = json.load(file)

        
        # Reconstruye las estructuras utilizando el método from_dict
        for structure_data in data.get('structures', []):
            structure = Structure.from_dict(structure_data)
            #structure.calculate_damage_per_structure()  # Añadir esta línea para calcular el daño por explosivo
            raid_calculator.add_structure(structure)
        
        # Reconstruye los explosivos utilizando el método from_dict
        for explosive_data in data.get('explosives', []):
            explosive = Explosive.from_dict(explosive_data)
            raid_calculator.add_explosive(explosive)
        
        # Establece la configuración
        raid_calculator.config = data.get('config', raid_calculator.config)
    
    except FileNotFoundError:
        print("No se encontró el archivo de datos. Se creará un nuevo archivo al guardar.")

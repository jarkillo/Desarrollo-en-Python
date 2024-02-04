import itertools # itertools sirve para generar combinaciones de elementos
import pulp # pip install pulp # pulp sirve para resolver problemas de optimización lineal
import json # json sirve para guardar y cargar datos en formato JSON
import os # os sirve para comprobar si existe un archivo
import logging # logging sirve para imprimir mensajes de depuración

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Structure:
    def __init__(self, name, hp, required_explosives):
        self.name = name
        self.hp = hp
        self.required_explosives = required_explosives

    def calculate_average_damage(self, explosives):
        for explosive_name, details in self.required_explosives.items():
            # Encuentra el explosivo correspondiente en la lista proporcionada
            explosive = next((exp for exp in explosives if exp.name == explosive_name), None)
            if explosive and details['quantity'] > 0:
                details['average_damage'] = self.hp / details['quantity']
    
    def calculate_damage_per_structure(self, explosives):
        for explosive_name, details in self.required_explosives.items():
            explosive = next((exp for exp in explosives if exp.name == explosive_name), None)
            if explosive and details['quantity'] > 0:
                explosive.damage_per_structure[self.name] = self.hp / details['quantity']


class Explosive:
    def __init__(self, name, sulfur_cost, gunpowder_cost, crafting_time):
        self.name = name
        self.sulfur_cost = sulfur_cost
        self.gunpowder_cost = gunpowder_cost
        self.crafting_time = crafting_time
        self.damage_per_structure = {}  # Un diccionario para almacenar el daño por estructura


class RaidCalculator:
    def __init__(self):
        self.structures = []
        self.explosives = []
        self.config = {
            "optimize_for": "sulfur",  # Puede ser "sulfur", "gunpowder", "crafting_time"
            "exclude_explosives": set(),
            "max_explosives": {}
        }

    def add_explosive(self, explosive):
        self.explosives.append(explosive)

    def set_optimization_criteria(self, criteria):
        if criteria in ["sulfur", "gunpowder", "crafting_time"]:
            self.config["optimize_for"] = criteria
        else:
            raise ValueError("Invalid optimization criteria. Choose 'sulfur', 'gunpowder', or 'crafting_time'.")

    def exclude_explosive(self, explosive_name):
        self.config["exclude_explosives"].add(explosive_name)

    def set_max_explosives(self, explosive_name, max_count):
        self.config["max_explosives"][explosive_name] = max_count

    def add_structure(self, structure):

        self.structures.append(structure)

    def calculate_optimal_combination(self):
        structure_name = input("Introduce el nombre de la estructura para calcular la combinación óptima: ")
        combination, cost = self.find_optimal_combination(structure_name)
        print(f"Combinación óptima: {combination}")
        print(f"Coste total: {cost}")
    
    def find_optimal_combination(self, structure_name): # Linear Programming
        # Encuentra la estructura por su nombre
        structure = next((s for s in self.structures if s.name == structure_name), None)
        if not structure:
            raise ValueError(f"No structure found with the name {structure_name}")

        # Define el problema de optimización
        prob = pulp.LpProblem("RaidCostMinimization", pulp.LpMinimize)

        # Define las variables de decisión
        explosives_vars = {explosive.name: pulp.LpVariable(f'num_{explosive.name}', 0, cat='Integer')
                        for explosive in raid_calculator.explosives}

        # Agrega la función objetivo
        if raid_calculator.config["optimize_for"] == "sulfur":
            prob += pulp.lpSum([explosive.sulfur_cost * explosives_vars[explosive.name]
                                for explosive in raid_calculator.explosives])
        elif raid_calculator.config["optimize_for"] == "gunpowder":
            prob += pulp.lpSum([explosive.gunpowder_cost * explosives_vars[explosive.name]
                                for explosive in raid_calculator.explosives])
        elif raid_calculator.config["optimize_for"] == "crafting_time":
            prob += pulp.lpSum([explosive.crafting_time * explosives_vars[explosive.name]
                                for explosive in raid_calculator.explosives])
            
        # ... Agrega condiciones para otros criterios de optimización
        
        # Agrega las restricciones
        # Restricción de daño: el total de daño debe ser mayor o igual al HP de la estructura
        # Asegúrate de que la referencia a la HP de la estructura sea correcta
        # Suponiendo que la estructura ya está definida en este contexto
        prob += pulp.lpSum([explosives_vars[explosive.name] * explosive.damage_per_structure[structure_name]
                            for explosive in self.explosives if structure_name in explosive.damage_per_structure])

        # Restricciones de uso máximo de explosivos
        for explosive_name, max_count in raid_calculator.config["max_explosives"].items():
            if explosive_name in explosives_vars:
                prob += explosives_vars[explosive_name] <= max_count
        
        # Restricciones de exclusión de explosivos
        for explosive_name in raid_calculator.config["exclude_explosives"]:
            if explosive_name in explosives_vars:
                prob += explosives_vars[explosive_name] == 0

        # Resuelve el problema
        prob.solve()

        # Extrae la solución óptima
        optimal_combination = {explosive.name: explosives_vars[explosive.name].varValue
                            for explosive in raid_calculator.explosives
                            if explosives_vars[explosive.name].varValue > 0}

        # Extrae el costo óptimo
        optimal_cost = pulp.value(prob.objective)

        return optimal_combination, optimal_cost
    
    def calculate_raid_cost(self):
        total_cost = {
            'sulfur': 0,
            'gunpowder': 0,
            'crafting_time': 0
    }
    
        for structure in self.structures:
            for explosive_name, details in structure.required_explosives.items():
                explosive = next((exp for exp in self.explosives if exp.name == explosive_name), None)
                if explosive:
                    total_cost['sulfur'] += details['quantity'] * explosive.sulfur_cost
                    total_cost['gunpowder'] += details['quantity'] * explosive.gunpowder_cost
                    total_cost['crafting_time'] += details['quantity'] * explosive.crafting_time
        
        return total_cost

def add_structure_user_interface(raid_calculator):
    print("\n--- Añadir Nueva Estructura ---")
    
    try:
        name = input("Introduce el nombre de la estructura: ").strip()
        hp = int(input("Introduce los HP de la estructura: ").strip())
    except ValueError:
        logging.error("La cantidad de HP debe ser un número entero.")
        return

    hp = int(hp)
    required_explosives = {}
    print("\nIntroduce la cantidad de cada explosivo necesario:")
    for explosive in raid_calculator.explosives:
        quantity_str = input(f"- {explosive.name}: ").strip()
        if not quantity_str.isdigit():
            print(f"La cantidad para {explosive.name} debe ser un número entero.")
            return

        quantity = int(quantity_str)
        required_explosives[explosive.name] = {"quantity": quantity}

    new_structure = Structure(name, hp, required_explosives)
    raid_calculator.add_structure(new_structure)
    print(f"La estructura '{name}' ha sido añadida con éxito.\n")


def add_explosive_user_interface(raid_calculator):
    print("\n--- Añadir Nuevo Explosivo ---")
    name = input("Introduce el nombre del explosivo: ").strip()
    sulfur_cost_str = input("Introduce el coste de azufre del explosivo: ").strip()
    gunpowder_cost_str = input("Introduce el coste de pólvora del explosivo: ").strip()
    crafting_time_str = input("Introduce el tiempo de fabricación (en segundos) del explosivo: ").strip()
    
    if not (sulfur_cost_str.isdigit() and gunpowder_cost_str.isdigit() and crafting_time_str.isdigit()):
        print("El coste de azufre, el coste de pólvora y el tiempo de fabricación deben ser números enteros.")
        return

    sulfur_cost = int(sulfur_cost_str)
    gunpowder_cost = int(gunpowder_cost_str)
    crafting_time = int(crafting_time_str)



    new_explosive = Explosive(name, sulfur_cost, gunpowder_cost, crafting_time)
    raid_calculator.add_explosive(new_explosive)
    print(f"El explosivo '{name}' ha sido añadido con éxito.\n")


def set_optimization_criteria_user_interface(raid_calculator):
    print("\n--- Establecer Criterio de Optimización ---")
    criteria_list = ["sulfur", "gunpowder", "crafting_time"]
    criteria = select_from_list(criteria_list, "Selecciona un criterio: ")
    raid_calculator.set_optimization_criteria(criteria)
    print(f"Criterio de optimización establecido a '{criteria}'.\n")


def exclude_explosive_user_interface(raid_calculator):
    print("\n--- Excluir Explosivos ---")
    explosives_list = [explosive.name for explosive in raid_calculator.explosives if explosive.name not in raid_calculator.config['exclude_explosives']]
    if not explosives_list:
        print("No hay explosivos disponibles para excluir.")
        return
    selected_explosive = select_from_list(explosives_list, "Selecciona un explosivo para excluir: ")
    raid_calculator.exclude_explosive(selected_explosive)
    print(f"El explosivo '{selected_explosive}' ha sido excluido.\n")

def select_from_list(options, prompt):
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    while True:
        selection = input(prompt).strip()
        if not selection.isdigit() or not (0 < int(selection) <= len(options)):
            print("Selección inválida. Introduce un número de la lista.")
        else:
            return options[int(selection) - 1]


def set_max_explosives_user_interface(raid_calculator):
    print("\n--- Establecer Número Máximo de Explosivos ---")
    explosives_list = [explosive.name for explosive in raid_calculator.explosives]
    selected_explosive = select_from_list(explosives_list, "Selecciona un explosivo para limitar: ")
    max_count_str = input(f"Introduce el número máximo de {selected_explosive}: ").strip()
    if not max_count_str.isdigit():
        print("La cantidad máxima debe ser un número entero.")
        return

    max_count = int(max_count_str)
    raid_calculator.set_max_explosives(selected_explosive, max_count)
    print(f"El número máximo de '{selected_explosive}' se ha establecido en {max_count}.\n")


def calculate_optimal_combination_user_interface(raid_calculator):
    print("\n--- Calcular Combinación Óptima ---")
    structures_list = [structure.name for structure in raid_calculator.structures]
    selected_structure = select_from_list(structures_list, "Selecciona una estructura para calcular la combinación óptima: ")
    optimal_combination, optimal_cost = raid_calculator.find_optimal_combination(selected_structure)
    print(f"La combinación óptima para destruir '{selected_structure}' es: {optimal_combination} con un costo de {optimal_cost}\n")


def save_data(raid_calculator):
    # Estructura de los datos a ser guardados
    data = {
        "structures": [
            {
                "name": structure.name,
                "hp": structure.hp,
                "required_explosives": structure.required_explosives
            } for structure in raid_calculator.structures
        ],
        "explosives": [
            {
                "name": explosive.name,
                "sulfur_cost": explosive.sulfur_cost,
                "gunpowder_cost": explosive.gunpowder_cost,
                "crafting_time": explosive.crafting_time
                # Asegúrate de que no hay más atributos necesarios aquí
            } for explosive in raid_calculator.explosives
        ],
        "config": {
            "optimize_for": raid_calculator.config["optimize_for"],
            "exclude_explosives": list(raid_calculator.config["exclude_explosives"]),  # Convertimos el conjunto a lista
            "max_explosives": raid_calculator.config["max_explosives"]
        }
    }
    
    # Guardar los datos en un archivo JSON
    with open('raid_calculator_data.json', 'w') as file:
        json.dump(data, file, indent=4)


def load_data(raid_calculator, filename='raid_calculator_data.json'):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            logging.info("Datos cargados desde el archivo JSON:")

            raid_calculator.structures = [Structure(**struct) for struct in data['structures']]
            raid_calculator.explosives = [Explosive(**exp) for exp in data['explosives']]
            raid_calculator.config = data['config']
            logging.info("Estructuras después de la carga:", raid_calculator.structures)
            logging.info("Explosivos después de la carga:", raid_calculator.explosives)

    except FileNotFoundError:
        print(f"No se encontró el archivo de datos {filename}. Se iniciará una nueva sesión.")
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")


# Esta función ejecuta el menú principal del programa
def main_menu(raid_calculator):
    while True:
        print("\n--- Menú Principal de la Calculadora de Raideos en Rust ---")
        print("1. Añadir estructura")
        print("2. Añadir explosivo")
        print("3. Establecer criterio de optimización")
        print("4. Excluir explosivos")
        print("5. Establecer número máximo de un tipo de explosivo")
        print("6. Calcular combinación óptima de explosivos")
        print("7. Salir")

        choice = input("Elige una opción: ").strip()

        data_changed = False  # Flag para rastrear si los datos han cambiado

        if choice == '1':
            add_structure_user_interface(raid_calculator)
            data_changed = True
        elif choice == '2':
            add_explosive_user_interface(raid_calculator)
            data_changed = True
        elif choice == '3':
            set_optimization_criteria_user_interface(raid_calculator)
            data_changed = True
        elif choice == '4':
            exclude_explosive_user_interface(raid_calculator)
            data_changed = True
        elif choice == '5':
            set_max_explosives_user_interface(raid_calculator)
            data_changed = True
        elif choice == '6':
            calculate_optimal_combination_user_interface(raid_calculator)
        elif choice == '7':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida, intenta de nuevo.")
        
        if data_changed:
            save_data(raid_calculator)  # Solo guarda los datos si han cambiado


# Punto de entrada principal del programa
if __name__ == "__main__":
    raid_calculator = RaidCalculator()
    if os.path.exists('raid_calculator_data.json'):
        load_data(raid_calculator)
    main_menu(raid_calculator)

# Para ejecutar el programa, ejecuta el siguiente comando en la terminal:
# python raidcalculator.py

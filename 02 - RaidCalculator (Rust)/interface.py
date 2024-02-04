import os

# imports necesarios
from operations import find_optimal_combination
from classes import Structure, Explosive, RaidCalculator
from typing import List, Dict
from data_manager import save_data, load_data

def main_menu(calculator: RaidCalculator):
    # Cargar datos al inicio
    load_data(calculator)
    # Flag para rastrear si los datos han cambiado
    data_changed = False
    # Modo administrador desactivado por defecto
    admin_mode = False

    while True:
        print("\n--- Raid Calculator Main Menu ---")
        
        # Si el modo de administrador está activado, mostrar opciones de administrador
        if admin_mode:
            print("1. Añadir Estructura")
            print("2. Añadir Explosivo")
            print("3. Calcular Combinación Óptima de Raid")
            print("4. Configurar filtros")
            print("5. Desactivar modo Admin (Ahora mismo ON)")
            print("6. Exit")
        else:
            print("1. Calcular Combinación Óptima de Raid")
            print("2. Set Configuration")
            print("3. Activar modo Amin (Ahora mismo OFF) (Ojo, puedes hacer que el programa no funcione correctamente)")
            print("4. Exit")
        
        choice = input("Choose an option: ")

        if admin_mode:
            if choice == '1':
                add_structure_user_interface(calculator)
                data_changed = True
            elif choice == '2':
                add_explosive_user_interface(calculator)
                data_changed = True
            elif choice == '3':
                calculate_optimal_combination_user_interface(calculator)
            elif choice == '4':
                set_config(calculator)
                data_changed = True
            elif choice == '5':
                # Alternar el modo de administrador
                admin_mode = not admin_mode
            elif choice == '6':
                print("Hasta pronto.")
                break
            else:
                print("Opcion incorrecta, intenta de nuevo.")
        else:
            if choice == '1':
                calculate_optimal_combination_user_interface(calculator)
            elif choice == '2':
                set_config(calculator)
                data_changed = True
            elif choice == '3':
                # Alternar el modo de administrador
                admin_mode = not admin_mode
            elif choice == '4':
                print("Hasta pronto.")
                break
            else:
                print("Opcion incorrecta, intenta de nuevo.")
        
        # Guardar datos si han cambiado
        if data_changed:
            save_data(calculator)


# Añade una estructura al calculador
def add_structure_user_interface(raid_calculator):
    print("\n--- Añadir Nueva Estructura ---")
    
    name = input("Introduce el nombre de la estructura: ").strip()
    hp_str = input("Introduce los HP de la estructura: ").strip()
    if not hp_str.isdigit():
        print("La cantidad de HP debe ser un número entero.")
        return
    hp = int(hp_str)

    required_explosives = {}
    damage_per_explosive = {}
    print("\nIntroduce la cantidad de cada explosivo necesario y su daño correspondiente:")
    for explosive in raid_calculator.explosives:
        while True:
            quantity_str = input(f"- {explosive.name} (cantidad, 0 si no es necesario): ").strip()
            if quantity_str.isdigit():
                quantity = int(quantity_str)
                if quantity > 0:  # Solo añadir si se necesita al menos uno
                    required_explosives[explosive.name] = {"quantity": quantity}
                    # Pedir al usuario que introduzca el daño que ese explosivo causa a la estructura
                    while True:
                        damage_str = input(f"  - Daño de {explosive.name} a la estructura: ").strip()
                        if damage_str.replace('.', '', 1).isdigit():
                            damage = float(damage_str)
                            damage_per_explosive[explosive.name] = damage
                            break
                        else:
                            print("El daño debe ser un número. Inténtalo de nuevo.")
                break
            else:
                print(f"La cantidad para {explosive.name} debe ser un número entero. Inténtalo de nuevo.")
    
    # Crear la nueva estructura con el daño por explosivo proporcionado
    new_structure = Structure(name, hp, required_explosives, damage_per_explosive)
    
    # Añadir la nueva estructura al calculador
    raid_calculator.add_structure(new_structure)
    print(f"La estructura '{name}' ha sido añadida con éxito. Daño por explosivo actualizado.\n")




# Añade un explosivo al calculador

def add_explosive_user_interface(raid_calculator):
    print("\n--- Añadir Nuevo Explosivo ---")
    name = input("Introduce el nombre del explosivo: ").strip()
    
    materials_cost = {}
    while True:
        material = input("Introduce el nombre del material (o 'salir' para terminar): ").strip()
        if material.lower() == 'salir':
            break
        cost_str = input(f"Introduce el coste de {material}: ").strip()
        if not cost_str.isdigit():
            print(f"El coste de {material} debe ser un número entero.")
            continue
        materials_cost[material] = int(cost_str)
    
    crafting_time_str = input("Introduce el tiempo de fabricación (en segundos) del explosivo: ").strip()
    if not crafting_time_str.isdigit():
        print("El tiempo de fabricación debe ser un número entero.")
        return

    crafting_time = int(crafting_time_str)
    new_explosive = Explosive(name, materials_cost, crafting_time)
    raid_calculator.add_explosive(new_explosive)
    print(f"El explosivo '{name}' ha sido añadido con éxito.\n")



def calculate_optimal_combination_user_interface(raid_calculator):
    print("\n--- Calcular Combinación Óptima ---")
    structure_quantities = {}
    while True:
        print("\nEstructuras disponibles:")
        for i, structure in enumerate(raid_calculator.structures, start=1):
            print(f"{i}. {structure.name}")
        print(f"{len(raid_calculator.structures) + 1}. Calcular óptimos")

        choice = input("Selecciona una estructura o calcula óptimos: ").strip()

        # Verificar si la elección es para calcular los óptimos
        if choice.isdigit() and int(choice) == len(raid_calculator.structures) + 1:
            break

        # Si se ha seleccionado una estructura, pedir la cantidad
        if choice.isdigit() and 0 < int(choice) <= len(raid_calculator.structures):
            selected_structure = raid_calculator.structures[int(choice) - 1]
            quantity = input(f"¿Cuántas '{selected_structure.name}'? ").strip()
            if quantity.isdigit() and int(quantity) > 0:
                structure_quantities[selected_structure.name] = int(quantity)
            else:
                print("Por favor, introduce un número entero válido.")
        else:
            print("Selección inválida. Introduce un número de la lista.")

    if not structure_quantities:
        print("No se ha ingresado ninguna estructura para calcular.")
        return

    def remove_max_explosive(self, explosive_name):
        self.config["max_explosives"].pop(explosive_name, None)  # Elimina el explosivo de la configuración si existe



# Si se han ingresado estructuras y cantidades, proceder con el cálculo
    optimal_combination, total_explosives, total_costs = find_optimal_combination(
        raid_calculator.structures, 
        structure_quantities, 
        raid_calculator.explosives, 
        raid_calculator.config
    )

    # Muestra la combinación óptima y los costos asociados para cada estructura
    print ("\n--- Resultados del cálculo ---")
    print ("=====================================")

    print("\nLa combinación óptima de explosivos para cada estructura es:")
    for structure_name, explosives in optimal_combination.items():
        # Aquí debes asegurarte de obtener la cantidad correcta para cada estructura
        quantity = structure_quantities.get(structure_name, "Cantidad no especificada")
        print(f"\n{structure_name} x {quantity}:")
        if isinstance(explosives, dict) and explosives:
            for explosive_name, quantity in explosives.items():
                print(f"  {explosive_name}: {quantity}")
        else:
            print("  No se encontró una solución factible.")

    # Muestra la suma total de explosivos necesarios para todas las estructuras
    print("\nCombinación óptima total para todas las estructuras seleccionadas:")
    for explosive_name, quantity in total_explosives.items():
        print(f"{explosive_name}: {quantity}")

    # Espera a que el usuario presione Enter para continuar
    input("Presiona Enter para continuar...")

    # Muestra los costos totales de materiales y tiempo de fabricación
    print("\nCostos totales de materiales y tiempo de fabricación para la combinación óptima:")
    print("===============================================================================")

    print ("Material: Cantidad")
    print ("------------------")
    print ("\n")
    for material, cost in total_costs['materials'].items():
        print(f"{material}: {cost}")
    print(f"\nTiempo total de fabricación: {total_costs['crafting_time']} segundos")

 # Espera a que el usuario presione Enter para continuar
    input("Presiona Enter para continuar...")

def set_config(calculator: RaidCalculator):
    print("\n--- Configuración del Calculador de Raids ---")

    # Establecer el criterio de optimización
    print("1. Establecer criterio de optimización")
    # Excluir explosivos
    print("2. Excluir explosivos")
    # Establecer el número máximo de un tipo de explosivo
    print("3. Establecer número máximo de un tipo de explosivo")
    print("4. Volver al menú principal")

    choice = input("Selecciona una opción de configuración: ").strip()

    if choice == '1':
        criteria_list = calculator.get_valid_criteria()  # Obtén los criterios basados en los explosivos actuales
        if criteria_list:  # Verifica que la lista no esté vacía
            print("Criterios de optimización disponibles: ", criteria_list)
            criteria = input("Selecciona un criterio para la optimización: ").strip()
            if criteria in criteria_list:
                calculator.set_optimization_criteria(criteria)
                print(f"Criterio de optimización establecido a '{criteria}'.")
            else:
                print("Opción no válida, por favor elige un criterio de la lista.")
        else:
            print("No hay criterios de optimización disponibles. Asegúrate de haber añadido explosivos con materiales.")

    elif choice == '2':
        explosive_list = [explosive.name for explosive in calculator.explosives]
        print("Explosivos disponibles: ", explosive_list)
        explosive_to_toggle = input("Selecciona un explosivo para excluir o incluir: ").strip()

        if explosive_to_toggle in explosive_list:
            if explosive_to_toggle in calculator.config["exclude_explosives"]:
                calculator.exclude_explosive(explosive_to_toggle)
                print(f"Explosivo '{explosive_to_toggle}' incluido nuevamente en los cálculos.")
            else:
                calculator.exclude_explosive(explosive_to_toggle)
                print(f"Explosivo '{explosive_to_toggle}' excluido de los cálculos.")
        else:
            print("Explosivo no válido.")

    elif choice == '3':
        explosive_list = [explosive.name for explosive in calculator.explosives]
        print("Explosivos disponibles para establecer un máximo: ", explosive_list)
        explosive_to_limit = input("Selecciona un explosivo para limitar o escribe 'reset' para eliminar el límite de uno: ").strip()
        
        # Opción para resetear el límite de un explosivo
        if explosive_to_limit.lower() == 'reset':
            explosive_to_reset = input("Escribe el nombre del explosivo para eliminar su límite: ").strip()
            if explosive_to_reset in calculator.config["max_explosives"]:
                del calculator.config["max_explosives"][explosive_to_reset]
                print(f"Se ha eliminado el límite para el explosivo '{explosive_to_reset}'.")
            else:
                print("No se encontró un límite establecido para ese explosivo o el nombre es incorrecto.")
        elif explosive_to_limit in explosive_list:
            max_count = input(f"Introduce el número máximo de {explosive_to_limit} o escribe 'none' para no tener límite: ").strip()
            if max_count.isdigit() and int(max_count) > 0:
                calculator.set_max_explosives(explosive_to_limit, int(max_count))
                print(f"Número máximo de '{explosive_to_limit}' establecido en {max_count}.")
            elif max_count.lower() == 'none':  # Opción para eliminar el límite
                calculator.remove_max_explosive(explosive_to_limit)
                print(f"Se ha eliminado el límite para el explosivo '{explosive_to_limit}'.")
            else:
                print("Por favor, introduce un número entero positivo válido o 'none' para no tener límite.")
        else:
            print("Explosivo no válido.")



    elif choice == '4':
        return
    else:
        print("Opción no válida, intenta de nuevo.")


if __name__ == "__main__":
    raid_calculator = RaidCalculator()
    main_menu(raid_calculator)

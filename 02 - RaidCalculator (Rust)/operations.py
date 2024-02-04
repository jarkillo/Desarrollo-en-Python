import pulp
from typing import List, Dict, Tuple
from classes import Explosive, Structure

# Función que calcula la combinación óptima de explosivos para una configuración dada
def find_optimal_combination(structures, quantities, explosives, config):
    # Inicializa los diccionarios que contendrán las soluciones y costos
    unique_solutions_by_structure = {}
    total_explosives = {}
    total_costs = {'materials': {}, 'crafting_time': 0}
    
    # Crea un diccionario de factores de costo para cada material
    cost_factors = {
        material: (lambda exp, m=material: exp.materials_cost.get(m, 0))
        for material in get_unique_materials(explosives)
    }
    
    # Función objetivo de costos solo para el material seleccionado para la optimización
    cost_func = cost_factors[config['optimize_for']]

    # Inicializa un diccionario para llevar un seguimiento del total de cada explosivo utilizado
    global_explosive_count = {explosive.name: 0 for explosive in explosives}

    # Resuelve un problema de optimización para cada estructura única
    for structure_name, quantity in quantities.items():
        prob = pulp.LpProblem(f"RaidCostMinimization_{structure_name}", pulp.LpMinimize)
        
        # Filtra los explosivos excluidos
        filtered_explosives = [
            explosive for explosive in explosives
            if explosive.name not in config.get("exclude_explosives", [])
        ]

        # Variables de decisión
        explosives_vars = {
            explosive.name: pulp.LpVariable(f'num_{explosive.name}_{structure_name}',
                                            lowBound=0, cat='Integer')
            for explosive in filtered_explosives
        }
        
        # Encuentra la estructura correspondiente
        structure = next((s for s in structures if s.name == structure_name), None)
        if not structure:
            print(f"Estructura {structure_name} no encontrada en la lista de estructuras.")
            continue
        
        # Restricción de daño
        total_hp = structure.hp
        damage_constraint = pulp.lpSum(
            explosives_vars[explosive.name] * structure.damage_per_explosive.get(explosive.name, 0)
            for explosive in filtered_explosives
        )
        prob += damage_constraint >= total_hp

        # Función objetivo
        prob += pulp.lpSum(
            cost_func(explosive) * explosives_vars[explosive.name]
            for explosive in filtered_explosives
        )

        # Restricción para el máximo de cada tipo de explosivo
        for explosive_name, max_count in config.get("max_explosives", {}).items():
            if explosive_name in explosives_vars:
                # La cantidad usada de cada explosivo no debe exceder el máximo global menos lo que ya se ha utilizado
                prob += explosives_vars[explosive_name] * quantity <= max_count - global_explosive_count[explosive_name]

        # Resuelve el problema
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Verifica la solución
        if prob.status == pulp.LpStatusInfeasible:
            print(f"No se encontró una solución factible para {structure_name}.")
            unique_solutions_by_structure[structure_name] = "No se encontró una solución factible"
        else:
            # Solución única por estructura
            unique_solution = {
                explosive.name: int(explosives_vars[explosive.name].varValue)
                for explosive in filtered_explosives
                if explosives_vars[explosive.name].varValue > 0
            }
            unique_solutions_by_structure[structure_name] = unique_solution
            
            # Actualiza el conteo global de explosivos usados
            for explosive_name, amount in unique_solution.items():
                global_explosive_count[explosive_name] += amount * quantity
                total_explosives[explosive_name] = total_explosives.get(explosive_name, 0) + amount * quantity

            # Costo total
            for explosive in filtered_explosives:
                var_value = explosives_vars[explosive.name].varValue
                if var_value > 0:
                    for material, cost in explosive.materials_cost.items():
                        total_costs['materials'][material] = total_costs['materials'].get(material, 0) + cost * var_value * quantity
                    total_costs['crafting_time'] += explosive.crafting_time * var_value * quantity

    return unique_solutions_by_structure, total_explosives, total_costs


def calculate_raid_cost(structures, explosives, quantities, material_costs):
    total_costs = {material: 0 for material in material_costs}  # 'material' debe ser una lista de materiales posibles
    for structure_name, quantity in quantities.items():
        structure = next((s for s in structures if s.name == structure_name), None)
        if structure:
            for explosive in explosives:
                if explosive.name in structure.damage_per_explosive:
                    for material, cost_per_explosive in explosive.materials_cost.items():
                        total_costs[material] += cost_per_explosive * quantity
    return total_costs


# Otras funciones auxiliares necesarias para el proceso de cálculo
def calculate_crafting_time(optimal_combination, explosives):
    total_crafting_time = 0
    for explosive_name, quantity in optimal_combination.items():
        explosive = next((exp for exp in explosives if exp.name == explosive_name), None)
        if explosive:
            total_crafting_time += explosive.crafting_time * quantity
    return total_crafting_time

def get_unique_materials(explosives):
    unique_materials = set()
    for explosive in explosives:
        unique_materials.update(explosive.materials_cost.keys())
    return list(unique_materials)

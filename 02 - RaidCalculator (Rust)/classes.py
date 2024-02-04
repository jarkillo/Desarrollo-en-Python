# imports

import pulp # pip install pulp # Sirve para optimizar el problema de minimización de costos

# classes.py

class Structure:
    def __init__(self, name, hp, required_explosives, damage_per_explosive=None):
        self.name = name
        self.hp = hp
        self.required_explosives = required_explosives
        self.damage_per_explosive = damage_per_explosive if damage_per_explosive else {}

    def set_damage_per_explosive(self, damage_dict):
        for explosive, damage in damage_dict.items():
            self.damage_per_explosive[explosive] = damage


# Métodos de la clase Structure


    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data['name'],
            hp=data['hp'],
            required_explosives=data['required_explosives'],
            damage_per_explosive=data['damage_per_explosive']
        )

    def to_dict(self):
    # Convierte la estructura a un diccionario para la serialización
        return {
            'name': self.name,
            'hp': self.hp,
            'required_explosives': self.required_explosives,
            'damage_per_explosive': self.damage_per_explosive
            }


# Clase para explosivos
class Explosive:
    def __init__(self, name, materials_cost, crafting_time):  # Constructor
        self.name = name  # Nombre del explosivo
        self.materials_cost = materials_cost  # Diccionario con los materiales necesarios para fabricar el explosivo
        self.crafting_time = crafting_time  # Tiempo de fabricación del explosivo


    # Método para calcular el daño por estructura
    #def add_structure_damage(self, structure_name, damage):
     #   self.damage_per_structure[structure_name] = damage


    @classmethod
    def from_dict(cls, data):
        explosive = cls(
            name=data['name'],
            materials_cost=data['materials_cost'],
            crafting_time=data['crafting_time']
        )
        #explosive.damage_per_structure = data.get('damage_per_structure', {})
        return explosive

    # Método para convertir el objeto en un diccionario
    def to_dict(self):
        return {
            'name': self.name,
            'materials_cost': self.materials_cost,
            'crafting_time': self.crafting_time,
            #'damage_per_structure': self.damage_per_structure
        }




# Clase para el calculador de costos de raid
class RaidCalculator:
    def __init__(self): # Constructor
        self.structures = [] # Lista de estructuras
        self.explosives = [] # Lista de explosivos
        self.config = {
            "optimize_for": "sulfur",  # Puede ser cualquier material, como "sulfur", "gunpowder", etc.
            "exclude_explosives": set(),  # Conjunto de nombres de explosivos a excluir
            "max_explosives": {}  # Diccionario de máximos por tipo de explosivo
        }

# Métodos de la clase RaidCalculator

    def to_dict(self):
        return {
            'structures': [structure.to_dict() for structure in self.structures],
            'explosives': [explosive.to_dict() for explosive in self.explosives],
            'config': self.config
        }
    

    # Método para agregar explosivos
    def add_explosive(self, explosive):
        # Si el explosivo no es una instancia de Explosive, levanta un ValueError
        if not isinstance(explosive, Explosive):
            # Levanta un ValueError
            raise ValueError("Must add an instance of Explosive.")
        # Agrega el explosivo a la lista de explosivos
        self.explosives.append(explosive)

    # Método para agregar estructuras
    def add_structure(self, structure):
        # Si la estructura no es una instancia de Structure, levanta un ValueError
        if not isinstance(structure, Structure):
            # Levanta un ValueError
            raise ValueError("Must add an instance of Structure.")
        # Agrega la estructura a la lista de estructuras
        self.structures.append(structure)

    # Método para configurar el criterio de optimización
    def set_optimization_criteria(self, criteria):
        valid_criteria = self.get_valid_criteria() # Obtiene los criterios válidos
        if criteria not in valid_criteria: # Si el criterio no es válido
            raise ValueError(f"Invalid optimization criteria. Choose from {valid_criteria}.") # Levanta un ValueError
        self.config["optimize_for"] = criteria # Configura el criterio de optimización

    # Método para excluir explosivos
    def exclude_explosive(self, explosive_name):
        if explosive_name in self.config["exclude_explosives"]:
            self.config["exclude_explosives"].remove(explosive_name)  # Elimina si ya está excluido
        else:
            self.config["exclude_explosives"].append(explosive_name)  # Agrega si no está excluido

    # Método para configurar el máximo de explosivos
    def set_max_explosives(self, explosive_name, max_count):
        # Si el explosivo no existe, levanta un ValueError
        if explosive_name not in [exp.name for exp in self.explosives]:
            # Levanta un ValueError
            raise ValueError(f"Explosive {explosive_name} does not exist.")
        # Si el máximo de explosivos es menor a 0, levanta un ValueError
        if max_count < 0:
            # Levanta un ValueError
            raise ValueError("Maximum count cannot be negative.")
        # Configura el máximo de explosivos
        self.config["max_explosives"][explosive_name] = max_count

    # Método para obtener los criterios válidos
    def get_valid_criteria(self):
        # Debe devolver una lista de criterios válidos basados en los materiales disponibles en los explosivos.
        # Por ejemplo, puede devolver ['sulfur', 'gunpowder'] si esos son los materiales comunes en todos los explosivos.
        materials = set()
        # Itera sobre los explosivos
        for explosive in self.explosives:
            # Agrega los materiales del explosivo al conjunto de materiales
            materials.update(explosive.materials_cost.keys())
            # Devuelve una lista de materiales
        return list(materials)
    
     


# Las instancias explosive se crean asi

# c4_materiales = {"sulfur": 100, "gunpowder": 50, "charcoal": 10}
# c4 = Explosive("C4", c4_materiales, crafting_time=30)
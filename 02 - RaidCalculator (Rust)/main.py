from interface import main_menu, load_data
from classes import RaidCalculator

if __name__ == "__main__":
    raid_calculator = RaidCalculator()  # Crea una instancia de RaidCalculator
    main_menu(raid_calculator)  # Inicia el menú principal con la instancia

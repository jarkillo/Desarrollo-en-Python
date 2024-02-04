import json

def guardar_scrap_data(scrap_data):
    with open("scrap_data.json", "w") as file:
        json.dump(scrap_data, file, indent=4)
    print("Datos guardados con éxito")

def cargar_scrap_data():
    try:
        with open("scrap_data.json", "r") as file:
            scrap_data = json.load(file)
    except FileNotFoundError:
        print("No se encontró el archivo de datos. Se creará uno nuevo al guardar.")
        scrap_data = {}
    finally:
        return scrap_data

def add_scrap_data(scrap_data):
    item = input("Ingrese el nombre del item: ")
    material1 = input("Ingrese el primer material: ")
    material1_amount = int(input("Ingrese la cantidad del primer material: "))
    material2 = input("Ingrese el segundo material: ")
    material2_amount = int(input("Ingrese la cantidad del segundo material: "))
    scrap_data[item] = {material1: material1_amount, material2: material2_amount}
    return scrap_data

def list_available_items(scrap_data):
    print("Items disponibles para reciclar:")
    for i, item in enumerate(scrap_data.keys()):
        print(f"{i+1}. {item}")

def scrap_calculation(scrap_data):
    list_available_items(scrap_data)
    items = []
    while True:
        item = input("Seleccione un item para reciclar (número o nombre): ")
        try:
            item = int(item)
            item = list(scrap_data.keys())[item - 1]
        except (ValueError, IndexError):
            pass

        item_amount = int(input(f"Cantidad de {item} a reciclar: "))
        items.append((item, item_amount))

        add_more = input("¿Desea añadir más items? (s/n): ")
        if add_more.lower() == "n":
            break

    total_materials = {}
    for item, count in items:
        if item in scrap_data:
            for material, amount in scrap_data[item].items():
                if material not in total_materials:
                    total_materials[material] = 0
                total_materials[material] += amount * count

    print("Materiales totales:")
    for material, amount in total_materials.items():
        print(f"{material}: {amount}")

def scrap_calculator():
    scrap_data = cargar_scrap_data()
    print("Calculadora de scrap")
    print("====================")
    while True:
        print("1. Calcular scrap")
        print("2. Añadir datos")
        print("3. Salir")
        option = input("Ingrese una opción: ")

        if option == "1":
            scrap_calculation(scrap_data)
        elif option == "2":
            scrap_data = add_scrap_data(scrap_data)
            guardar_scrap_data(scrap_data)
        elif option == "3":
            print("Saliendo...")
            break
        else:
            print("Opción no válida")

def main():
    scrap_calculator()

if __name__ == "__main__":
    main()

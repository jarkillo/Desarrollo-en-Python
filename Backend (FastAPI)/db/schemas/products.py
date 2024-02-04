# Este fichero es para recoger lo que viene de base de datos y lo transforme a nuestro JSON (Diccionario)

def product_schema(product) -> dict:
    return {
        "product_id": str(product["_id"]),
        "product_name": product["product_name"],
        "product_category": product["product_category"]
    }


def products_schema(products) -> list:
    return [product_schema(product) for product in products]

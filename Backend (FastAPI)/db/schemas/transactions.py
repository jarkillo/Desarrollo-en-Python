# Este fichero es para recoger lo que viene de base de datos y transformarlo a nuestro JSON (Diccionario)

def transaction_schema(transaction) -> dict:
    return {
        "transaction_id": str(transaction["_id"]),
        "user_id": transaction["user_id"],
        "date": transaction["date"],
        "product_id": transaction["product_id"],
        "amount": transaction["amount"]
    }


def transactions_schema(transactions) -> list:
    return [transaction_schema(transaction) for transaction in transactions]

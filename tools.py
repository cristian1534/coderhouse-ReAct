from langchain_core.tools import tool


@tool
def buscar_pedidos(cliente_id: int) -> dict:
    """Busca en la base de datos los pedidos de un cliente y devuelve el total gastado.

    Utiliza esta herramienta siempre que el usuario pregunte por la cantidad de
    pedidos o el monto total de un cliente específico.

    Args:
        cliente_id: El número de identificación único del cliente (ejemplo: 102).
    """
    datos = {
        102: {"pedidos": 3, "total": 14500},
        103: {"pedidos": 1, "total": 3200},
    }
    if cliente_id in datos:
        return datos[cliente_id]
    return {"error": "Cliente no encontrado en los registros"}
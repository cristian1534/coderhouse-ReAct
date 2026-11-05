import asyncio
import json
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import buscar_pedidos

load_dotenv()


# 1. Estado con reducer add_messages
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 2. LLM Gemini con herramientas vinculadas
# Nota: los modelos Gemini 3.x no aceptan temperature ni parámetros de muestreo.
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
tools = [buscar_pedidos]
llm_with_tools = llm.bind_tools(tools)


# 3. Nodo del modelo
def call_model(state: AgentState):
    print("  [nodo agent] invocando Gemini...")
    response = llm_with_tools.invoke(state["messages"])
    print(f"  [nodo agent] respuesta: type={response.type} tool_calls={getattr(response, 'tool_calls', None)}")
    return {"messages": [response]}


# 4. Construcción del grafo (sin checkpointer todavía)
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")


# 5. Helpers para salida limpia y traza
def extraer_texto(mensaje) -> str:
    """Extrae el texto de un mensaje, soportando content como str o lista de bloques."""
    if isinstance(mensaje.content, str):
        return mensaje.content
    if isinstance(mensaje.content, list):
        return "".join(
            bloque.get("text", "") if isinstance(bloque, dict) else str(bloque)
            for bloque in mensaje.content
        )
    return str(mensaje.content)


def volcar_traza(resultado, nombre_archivo: str) -> None:
    """Vuelca los mensajes del estado a un JSON para evidencia de la traza ReAct."""
    traza = []
    for m in resultado["messages"]:
        traza.append({
            "tipo": m.type,
            "contenido": extraer_texto(m),
            "tool_calls": getattr(m, "tool_calls", None),
        })
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        json.dump(traza, f, ensure_ascii=False, indent=2)


# 6. Función de ejecución asíncrona
async def main():
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        config = {
            "configurable": {"thread_id": "sesion-102"},
            "recursion_limit": 10,
        }

        # Turno 1: el agente llama a buscar_pedidos(102)
        print("--- Turno 1 ---")
        print("→ Enviando pregunta a Gemini...")
        try:
            r1 = await asyncio.wait_for(
                graph.ainvoke(
                    {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?")]},
                    config=config,
                ),
                timeout=60,
            )
        except asyncio.TimeoutError:
            print("⏱️ Timeout: Gemini no respondió en 60 segundos.")
            return

        print("Respuesta:", extraer_texto(r1["messages"][-1]))
        volcar_traza(r1, "trace_turno1.json")

        # Turno 2: mismo thread_id, segunda invocación a la herramienta (cliente 103)
        print("\n--- Turno 2 (misma sesión) ---")
        try:
            r2 = await asyncio.wait_for(
                graph.ainvoke(
                    {"messages": [HumanMessage(content="¿Y cuántos pedidos tuvo el cliente 103?")]},
                    config=config,
                ),
                timeout=60,
            )
        except asyncio.TimeoutError:
            print("⏱️ Timeout en Turno 2.")
            return

        print("Respuesta:", extraer_texto(r2["messages"][-1]))
        volcar_traza(r2, "trace_turno2.json")


if __name__ == "__main__":
    asyncio.run(main())
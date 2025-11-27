from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import json




# 1. Configuración Inicial del modelo LLM
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)


# Preprocesador de Texto
def prepocess_text(text:str)->str:
    """Limpia el texto eliminando espacios extras y limitando longitud"""
    return text.strip()[:500]

preprocess = RunnableLambda(prepocess_text)


# Generador de Resúmenes
def generate_summary(llm: ChatOpenAI, text:str)->str:
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oración {text}"
    response = llm.invoke(prompt)
    return response.content



# Analizador de Sentimientos
def analyze_sentiment(llm: ChatOpenAI, text: str)->json:
    """Analiza el sentimiento y devuelve resultador estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
            Respondes ÚNICAMENTE en formato JSON válido:
            {
                {
                    "sentimiento": "positivo|negativo|neutro", 
                    "razón": "justificación breve"
                }
            
            }

            Texto: {text}
    """
    response = llm.invoke(prompt)
    try: 
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razón": "Error en análisis"}



# Función de Combinación
def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razón": data["sentimiento_data"]["razón"]
    }


# Función de Procesamiento Principal
def process_one(t):
    resumen = generate_summary(t)           # Llamada 1 al LLM
    sentimiento_data = analyze_sentiment(t) # Llamada 2 al LLM
    return merge_results({
        "resumen": resumen,
        "sentimiento_data": sentimiento_data
    }
)

# Convertir en Runnable
process = RunnableLambda(process_one)

# 7. Construcción de la Cadena Final
chain = preprocess | process


textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]

for texto in textos_prueba:
    resultado = chain.invoke(texto)
    print(f"Texto: {texto}")
    print(f"Resultado: {resultado}")
    print("-"*50)
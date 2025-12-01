from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv

load_dotenv()



# 1. Configuración Inicial del modelo LLM
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)


# Preprocesador de Texto
def prepocess_text(text:str)->str:
    """Limpia el texto eliminando espacios extras y limitando longitud"""
    return text.strip()[:500]

preprocess = RunnableLambda(prepocess_text)


# Generador de Resúmenes
def generate_summary(text:str)->str:
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oración {text}"
    response = llm.invoke(prompt)
    return response.content

summary_branch = RunnableLambda(generate_summary)

# Analizador de Sentimientos
def analyze_sentiment(text: str)->json:
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

sentiment_branch = RunnableLambda(analyze_sentiment)

# Función de Combinación
def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razón": data["sentimiento_data"]["razón"]
    }

merge_branch = RunnableLambda(merge_results)

# Función de Procesamiento Principal
# def process_one(t):
#     resumen = generate_summary(t)           # Llamada 1 al LLM
#     sentimiento_data = analyze_sentiment(t) # Llamada 2 al LLM
#     return merge_results({
#         "resumen": resumen,
#         "sentimiento_data": sentimiento_data
#     }
# )

# Convertir en Runnable
# process = RunnableLambda(process_one)

# 7. Construcción de la Cadena Final
# chain = preprocess | process


#Implementación de cadenas paralelas con RunnableParallel
parallel_branch = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": sentiment_branch,
})

# Implementación de la nueva cadena
chain = preprocess | parallel_branch | merge_branch



# textos_prueba = [
#     "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
#     "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
#     "El clima está nublado hoy, probablemente llueva más tarde."
# ]

# for texto in textos_prueba:
#     resultado = chain.invoke(texto)
#     print(f"Texto: {texto}")
#     print(f"Resultado: {resultado}")
#     print("-"*50)

# Prueba en Lote
reviews_batch = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]

#Implementación de procesamiento en lote
resultados = chain.batch(reviews_batch)

print(f"Resultados de procesamiento en lote: {resultados}")
import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process
from groq import Groq

app = Flask(__name__)

# Configuración de Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CARGA DEL HISTÓRICO (Memoria de Manuchar) ---
try:
    # Asegúrate de que el CSV esté en la misma carpeta que este archivo
    df_historico = pd.read_csv('historico_manuchar.csv')
    # Eliminamos duplicados para que la búsqueda sea más rápida y precisa
    df_historico = df_historico.drop_duplicates(subset=['Dscription', 'AcctCode'])
    historico_list = df_historico['Dscription'].astype(str).tolist()
    print("Histórico cargado correctamente.")
except Exception as e:
    print(f"Error cargando histórico: {e}")
    df_historico = None
    historico_list = []

def obtener_contexto_historico(desc_nueva):
    if not historico_list or df_historico is None:
        return "No hay datos históricos disponibles para esta empresa."
    
    # Buscamos las 3 descripciones más parecidas en los 42k registros
    matches = process.extract(desc_nueva, historico_list, limit=3)
    
    contexto = "\n--- MEMORIA CRÍTICA (HISTÓRICO DE MANUCHAR) ---\n"
    contexto += "En registros pasados, descripciones similares se categorizaron así:\n"
    
    for m in matches:
        # Buscamos la fila correspondiente al match
        fila = df_historico[df_historico['Dscription'] == m[0]].iloc[0]
        contexto += f"- Descripción: '{m[0]}' -> Cuenta usada: {fila['AcctCode']} ({fila['AcctName']})\n"
    
    contexto += "\nInstrucción: Si el histórico muestra una tendencia clara, prioriza ese criterio contable sobre la semántica general.\n"
    return contexto

def obtener_veredicto_ia(descripcion, cuenta_actual, opciones_reducidas, contexto_historico):
    prompt = f'''
Eres un Auditor de Sistemas Contables Senior experto en SAP Business One y el PCGE.
Tu misión es validar la cuenta contable de una factura de compra.

**DATOS ACTUALES:**
- Descripción del Ítem: "{descripcion}"
- Cuenta Asignada Actualmente: {cuenta_actual.get('AcctCode')} - {cuenta_actual.get('AcctName')}

**CONTEXTO DE LA EMPRESA (HISTÓRICO):**
{contexto_historico}

**CATÁLOGO DE CUENTAS DISPONIBLES (Sugerencias):**
{opciones_reducidas}

**REGLAS DE AUDITORÍA:**
1. PRIORIDAD HISTÓRICA: El equipo contable tiene criterios específicos (ej. Plastificado como Manipuleo). Si el histórico muestra una cuenta usada repetidamente para descripciones similares, súmalo a tu decisión.
2. CONSISTENCIA: Si la cuenta actual coincide con el histórico o con la semántica, marca 'es_correcta': true.
3. DETECCIÓN DE COMODINES: Si usan cuentas puente (Facturas por recibir, Cuentas por pagar) cuando existe una cuenta de gasto o activo específica, sugiere el cambio.
4. NIVEL DE DETALLE: Solo son válidas cuentas de registro (4-6 dígitos).

FORMATO DE SALIDA (JSON ESTRICTO):
{{
    "es_correcta": true/false,
    "codigo_sugerido": "XXXXXX",
    "nombre_sugerido": "XXXXXX",
    "justificacion": "Explicación técnica considerando el histórico de la empresa y la lógica contable."
}}
'''
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message.content
    except Exception as e:
        return str({"error_ia": str(e)})

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_sql = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        catalogo_completo = data.get('catalogo', [])

        if not catalogo_completo or not desc_sql:
            return jsonify({"error": "Faltan datos en la petición"}), 400

        # 1. Filtrar catálogo (Cuentas de registro)
        catalogo_detalle = [item for item in catalogo_completo if len(str(item.get('AcctCode', ''))) >= 4]
        
        # 2. Fuzzy Matching para reducir opciones del catálogo
        nombres_cat = [item['AcctName'] for item in catalogo_detalle]
        matches_cat = process.extract(desc_sql, nombres_cat, limit=10)
        top_10_nombres = [m[0] for m in matches_cat]
        opciones_reducidas = [item for item in catalogo_detalle if item['AcctName'] in top_10_nombres]

        # 3. Obtener el "aprendizaje" del histórico de 42k filas
        contexto_hist = obtener_contexto_historico(desc_sql)

        # 4. Veredicto final de la IA
        resultado_ia = obtener_veredicto_ia(desc_sql, cuenta_actual, opciones_reducidas, contexto_hist)
        
        return resultado_ia

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return "API de Auditoría Híbrida (IA + Histórico) activa."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

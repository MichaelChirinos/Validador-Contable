import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process
import json
from groq import Groq

app = Flask(__name__)

# Configuración de Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CARGA DEL HISTÓRICO (versión mejorada con más columnas) ---
try:
    # Cargamos el CSV con las nuevas columnas
    df_historico = pd.read_csv(
        'results.csv', 
        sep=';',  # Ajusta según tu separador real
        encoding='latin-1'
    )
    
    # Limpieza: quitamos filas que tengan la descripción vacía
    df_historico = df_historico.dropna(subset=['Dscription'])
    
    # Para la búsqueda fuzzy, seguimos usando SOLO la descripción
    historico_list = df_historico['Dscription'].astype(str).tolist()
    
    print(f"Histórico cargado: {len(historico_list)} filas encontradas.")
    print(f"Columnas disponibles: {df_historico.columns.tolist()}")
    
except Exception as e:
    print(f"Error crítico cargando histórico: {e}")
    df_historico = None
    historico_list = []

def obtener_contexto_historico(desc_nueva, proveedor=None, grupo_item=None, tipo_operacion=None, centro_costo=None):
    """
    Busca descripciones similares en el histórico (misma lógica de siempre)
    y agrega información complementaria si está disponible
    """
    if not historico_list or df_historico is None:
        return "No hay datos históricos disponibles para esta empresa."
    
    # Búsqueda fuzzy SOLO por descripción (como siempre funcionó)
    matches = process.extract(desc_nueva, historico_list, limit=3)
    
    contexto = "\n--- MEMORIA CRÍTICA (HISTÓRICO DE MANUCHAR) ---\n"
    contexto += "En registros pasados, descripciones similares se categorizaron así:\n"
    
    for m in matches:
        # Buscamos la fila correspondiente al match
        fila = df_historico[df_historico['Dscription'] == m[0]].iloc[0]
        contexto += f"- Descripción: '{m[0]}' -> Cuenta usada: {fila['AcctCode']} ({fila['AcctName']})\n"
    
    contexto += "\nInstrucción: Si el histórico muestra una tendencia clara, prioriza ese criterio contable sobre la semántica general.\n"
    
    # --- INFORMACIÓN COMPLEMENTARIA (no afecta la búsqueda, solo contexto) ---
    contexto += "\n--- INFORMACIÓN ADICIONAL DEL REGISTRO ACTUAL ---\n"
    if proveedor and pd.notna(proveedor):
        contexto += f"- Proveedor: {proveedor}\n"
    if grupo_item and pd.notna(grupo_item):
        contexto += f"- Grupo de Ítem: {grupo_item}\n"
    if tipo_operacion and pd.notna(tipo_operacion):
        contexto += f"- Tipo de Operación: {tipo_operacion}\n"
    if centro_costo and pd.notna(centro_costo):
        contexto += f"- Centro de Costo: {centro_costo}\n"
    
    contexto += "\nNota: Usa esta información adicional solo como referencia. La prioridad sigue siendo el histórico de descripciones similares.\n"
    
    return contexto

def obtener_veredicto_ia(descripcion, cuenta_actual, opciones_reducidas, contexto_historico):
    prompt = f'''
Eres un Auditor de Sistemas Contables Senior experto en SAP Business One y el PCGE.
Tu misión es validar la cuenta contable de una factura de compra.

**DATOS ACTUALES:**
- Descripción del Ítem: "{descripcion}"
- Cuenta Asignada Actualmente: {cuenta_actual.get('AcctCode')} - {cuenta_actual.get('AcctName')}

**CONTEXTO DE LA EMPRESA (HISTÓRICO + INFO ADICIONAL):**
{contexto_historico}

**CATÁLOGO DE CUENTAS DISPONIBLES (Sugerencias):**
{opciones_reducidas}

**REGLAS DE AUDITORÍA:**
1. PRIORIDAD HISTÓRICA: El equipo contable tiene criterios específicos. Si el histórico muestra una cuenta usada repetidamente para descripciones similares, prioriza ese criterio.
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
        return json.dumps({"error_ia": str(e)})

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_sql = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        catalogo_completo = data.get('catalogo', [])
        
        # Nuevos campos complementarios (opcionales)
        proveedor = data.get('proveedor', '')
        grupo_item = data.get('grupo_item', '')
        tipo_operacion = data.get('tipo_operacion', '')
        centro_costo = data.get('centro_costo', '')

        if not catalogo_completo or not desc_sql:
            return jsonify({"error": "Faltan datos en la petición"}), 400

        # 1. Filtrar catálogo (Cuentas de registro)
        catalogo_detalle = [item for item in catalogo_completo if len(str(item.get('AcctCode', ''))) >= 4]
        
        # 2. Fuzzy Matching para reducir opciones del catálogo (solo por nombre de cuenta)
        nombres_cat = [item['AcctName'] for item in catalogo_detalle]
        matches_cat = process.extract(desc_sql, nombres_cat, limit=10)
        top_10_nombres = [m[0] for m in matches_cat]
        opciones_reducidas = [item for item in catalogo_detalle if item['AcctName'] in top_10_nombres]

        # 3. Obtener el contexto histórico (con info complementaria)
        contexto_hist = obtener_contexto_historico(
            desc_sql, 
            proveedor=proveedor,
            grupo_item=grupo_item,
            tipo_operacion=tipo_operacion,
            centro_costo=centro_costo
        )

        # 4. Veredicto final de la IA
        resultado_ia = obtener_veredicto_ia(desc_sql, cuenta_actual, opciones_reducidas, contexto_hist)
        
        return resultado_ia, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return "API de Auditoría Híbrida (IA + Histórico) activa."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

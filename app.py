import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process
import json
from groq import Groq

app = Flask(__name__)

# Configuración de Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CARGA DEL HISTÓRICO MEJORADO ---
try:
    df_historico = pd.read_csv(
        'results.csv', 
        sep='\t',  # Asumo que es tabulación, si es coma cambia a ','
        encoding='latin-1'
    )
    
    # Limpieza: quitamos filas que tengan la descripción vacía
    df_historico = df_historico.dropna(subset=['Dscription'])
    
    # Creamos una columna combinada para mejor búsqueda
    df_historico['texto_busqueda'] = (
        df_historico['Dscription'].astype(str) + " " +
        df_historico['Proveedor'].astype(str) + " " +
        df_historico['ItmsGrpNam'].astype(str).fillna('')
    )
    
    historico_list = df_historico['texto_busqueda'].astype(str).tolist()
    print(f"Histórico cargado: {len(historico_list)} filas encontradas.")
    print(f"Columnas disponibles: {df_historico.columns.tolist()}")
    
except Exception as e:
    print(f"Error crítico cargando histórico: {e}")
    df_historico = None
    historico_list = []

def obtener_contexto_historico(desc_nueva, proveedor_nuevo=None):
    if not historico_list or df_historico is None:
        return "No hay datos históricos disponibles para esta empresa."
    
    # Construimos texto de búsqueda con descripción y proveedor
    texto_busqueda = desc_nueva
    if proveedor_nuevo:
        texto_busqueda += " " + proveedor_nuevo
    
    # Buscamos las 5 descripciones más parecidas
    matches = process.extract(texto_busqueda, historico_list, limit=5)
    
    contexto = "\n--- MEMORIA CRÍTICA (HISTÓRICO DE MANUCHAR) ---\n"
    contexto += "En registros pasados, descripciones similares se categorizaron así:\n"
    
    for m in matches:
        # Buscamos la fila correspondiente al match
        fila = df_historico[df_historico['texto_busqueda'] == m[0]].iloc[0]
        contexto += f"""
- Descripción: '{fila['Dscription']}'
  Proveedor: {fila['Proveedor']}
  Grupo Item: {fila['ItmsGrpNam'] if pd.notna(fila['ItmsGrpNam']) else 'N/A'}
  Tipo Operación: {fila['U_TipOper'] if pd.notna(fila['U_TipOper']) else 'N/A'}
  Cuenta usada: {fila['AcctCode']} ({fila['AcctName']})
  Centro Costo: {fila['OcrCode3'] if pd.notna(fila['OcrCode3']) else 'N/A'}
"""
    
    contexto += "\nInstrucción: Analiza patrones: mismo proveedor + misma descripción = misma cuenta contable.\n"
    return contexto

def obtener_veredicto_ia(descripcion, cuenta_actual, opciones_reducidas, contexto_historico, proveedor, grupo_item, tipo_operacion):
    prompt = f'''
Eres un Auditor de Sistemas Contables Senior experto en SAP Business One y el PCGE.
Tu misión es validar la cuenta contable de una factura de compra.

**DATOS ACTUALES:**
- Descripción del Ítem: "{descripcion}"
- Proveedor: {proveedor}
- Grupo de Ítem: {grupo_item if grupo_item else 'No especificado'}
- Tipo Operación: {tipo_operacion if tipo_operacion else 'No especificado'}
- Cuenta Asignada Actualmente: {cuenta_actual.get('AcctCode')} - {cuenta_actual.get('AcctName')}

**CONTEXTO DE LA EMPRESA (HISTÓRICO):**
{contexto_historico}

**CATÁLOGO DE CUENTAS DISPONIBLES (Sugerencias):**
{opciones_reducidas}

**REGLAS DE AUDITORÍA:**
1. PRIORIDAD HISTÓRICA: Si el histórico muestra una cuenta usada repetidamente para la misma combinación (descripción + proveedor), prioriza ese criterio.
2. PATRÓN DE PROVEEDOR: Mismo proveedor + mismo tipo de gasto = misma cuenta contable.
3. CONSISTENCIA: Si la cuenta actual coincide con el histórico o con la semántica, marca 'es_correcta': true.
4. DETECCIÓN DE COMODINES: Si usan cuentas puente (Facturas por recibir) cuando existe una cuenta específica, sugiere el cambio.

FORMATO DE SALIDA (JSON ESTRICTO):
{{
    "es_correcta": true/false,
    "codigo_sugerido": "XXXXXX",
    "nombre_sugerido": "XXXXXX",
    "justificacion": "Explicación técnica considerando el histórico del proveedor y la empresa."
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
        
        # Nuevos campos
        proveedor = data.get('proveedor', '')
        grupo_item = data.get('grupo_item', '')
        tipo_operacion = data.get('tipo_operacion', '')

        if not catalogo_completo or not desc_sql:
            return jsonify({"error": "Faltan datos en la petición"}), 400

        # 1. Filtrar catálogo (Cuentas de registro)
        catalogo_detalle = [item for item in catalogo_completo if len(str(item.get('AcctCode', ''))) >= 4]
        
        # 2. Fuzzy Matching para reducir opciones del catálogo
        nombres_cat = [item['AcctName'] for item in catalogo_detalle]
        matches_cat = process.extract(desc_sql, nombres_cat, limit=10)
        top_10_nombres = [m[0] for m in matches_cat]
        opciones_reducidas = [item for item in catalogo_detalle if item['AcctName'] in top_10_nombres]

        # 3. Obtener el "aprendizaje" del histórico con proveedor
        contexto_hist = obtener_contexto_historico(desc_sql, proveedor)

        # 4. Veredicto de IA
        resultado_ia = obtener_veredicto_ia(
            desc_sql, cuenta_actual, opciones_reducidas, 
            contexto_hist, proveedor, grupo_item, tipo_operacion
        )
        
        return resultado_ia, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return "API de Auditoría Híbrida (IA + Histórico) activa."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

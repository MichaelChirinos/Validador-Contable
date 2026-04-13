import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process, fuzz
from groq import Groq
import json
import re
from datetime import datetime

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- MEMORIA PERSISTENTE ---
MEMORIA_ARCHIVO = 'memoria.json'
FEEDBACK_ARCHIVO = 'feedback.json'

def cargar_memoria():
    if os.path.exists(MEMORIA_ARCHIVO):
        with open(MEMORIA_ARCHIVO, 'r', encoding='latin-1') as f:
            return json.load(f)
    return {}

def guardar_memoria(memoria):
    with open(MEMORIA_ARCHIVO, 'w', encoding='latin-1') as f:
        json.dump(memoria, f, indent=2)

def cargar_feedback():
    if os.path.exists(FEEDBACK_ARCHIVO):
        with open(FEEDBACK_ARCHIVO, 'r', encoding='latin-1') as f:
            return json.load(f)
    return {}

def guardar_feedback(feedback):
    with open(FEEDBACK_ARCHIVO, 'w', encoding='latin-1') as f:
        json.dump(feedback, f, indent=2)

# --- CARGA DE CATÁLOGO ---
df_catalogo = pd.read_csv(
    'results_1.csv', 
    sep=';',
    names=['AcctCode', 'AcctName'],
    encoding='latin-1',
    on_bad_lines='skip'
)
catalogo_dict = dict(zip(df_catalogo['AcctCode'].astype(str), df_catalogo['AcctName']))

def obtener_nombre_real(codigo):
    return catalogo_dict.get(str(codigo), None)

def extraer_palabras_clave(desc):
    """Extrae palabras clave relevantes"""
    if not desc or desc == 'NULL':
        return ""
    
    desc = desc.lower()
    # Eliminar patrones comunes
    desc = re.sub(r'dcc\s*\d+\s*-?\s*', '', desc)
    desc = re.sub(r'fee|ee', '', desc, flags=re.IGNORECASE)
    desc = re.sub(r'[a-z]+\s+[a-z]+\s+[a-z]+', '', desc)
    desc = re.sub(r'\d+', '', desc)
    desc = re.sub(r'[^\w\sáéíóúñ]', ' ', desc)
    
    palabras = re.findall(r'[a-záéíóúñ]{4,}', desc)
    return ' '.join(palabras[:3])

# --- CARGA DEL HISTÓRICO (con las columnas correctas) ---
print("📚 Cargando histórico...")
print("   Columnas: ItemCode, Dscription, AcctName, Proveedor, ItmsGrpNam, U_TipGasCos, U_TipOper, OcrCode3")

df_historico = pd.read_csv(
    'results.csv', 
    sep=';',  # Por el ejemplo que diste, usa punto y coma
    names=['ItemCode', 'Dscription', 'AcctName', 'Proveedor', 
           'ItmsGrpNam', 'U_TipGasCos', 'U_TipOper', 'OcrCode3'],
    encoding='latin-1', 
    on_bad_lines='skip'
)

# Limpiar datos NULL
df_historico = df_historico.replace('NULL', pd.NA)
df_historico = df_historico.dropna(subset=['Dscription', 'AcctName'])
df_historico = df_historico[df_historico['Dscription'].str.strip() != '']

print(f"📊 Histórico cargado: {len(df_historico)} descripciones")
print(f"   Ejemplo: '{df_historico.iloc[0]['Dscription'][:50]}' → {df_historico.iloc[0]['AcctName']}")

# Crear índice para búsqueda rápida
desc_to_acct = dict(zip(df_historico['Dscription'].str.lower(), df_historico['AcctName']))
historico_desc_list = df_historico['Dscription'].tolist()

# Cargar memorias
MEMORIA = cargar_memoria()
FEEDBACK_DB = cargar_feedback()
print(f"📚 Memoria: {len(MEMORIA)} patrones")
print(f"📝 Feedback: {len(FEEDBACK_DB)} correcciones")

def recordar(descripcion):
    """Busca coincidencia usando múltiples estrategias"""
    if not descripcion:
        return None
        
    desc_lower = descripcion.lower()
    
    # 1. Feedback manual (prioridad máxima)
    for key, value in FEEDBACK_DB.items():
        if key in desc_lower or desc_lower in key:
            print(f"   ✅ Feedback encontrado: {key[:30]}...")
            return {'codigo': value['codigo'], 'nombre': value['nombre'], 'score': 100, 'tipo': 'feedback'}
    
    # 2. Coincidencia exacta en histórico
    if desc_lower in desc_to_acct:
        nombre_historico = desc_to_acct[desc_lower]
        for code, name in catalogo_dict.items():
            if name == nombre_historico:
                print(f"   ✅ Coincidencia exacta")
                return {'codigo': code, 'nombre': name, 'score': 100, 'tipo': 'exacta'}
    
    # 3. Memoria por palabras clave
    patron = extraer_palabras_clave(descripcion)
    if patron and patron in MEMORIA:
        print(f"   📚 Memoria: {patron}")
        return MEMORIA[patron]
    
    # 4. Búsqueda por palabra clave en histórico
    palabras_busqueda = extraer_palabras_clave(descripcion).split()
    for palabra in palabras_busqueda:
        if len(palabra) > 3:
            matches = [d for d in historico_desc_list if palabra in d.lower()]
            if matches:
                mejor_match = matches[0]
                nombre_historico = desc_to_acct.get(mejor_match.lower())
                if nombre_historico:
                    for code, name in catalogo_dict.items():
                        if name == nombre_historico:
                            resultado = {'codigo': code, 'nombre': name, 'score': 70, 'tipo': 'palabra_clave'}
                            if patron:
                                MEMORIA[patron] = resultado
                                guardar_memoria(MEMORIA)
                            return resultado
    
    return None

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        codigo_actual = str(cuenta_actual.get('AcctCode', ''))
        nombre_actual = cuenta_actual.get('AcctName', '')
        
        print(f"\n📝 Auditando: {desc_f[:80]}")
        print(f"   Actual: {codigo_actual} - {nombre_actual}")
        
        recordado = recordar(desc_f)
        
        if recordado and recordado['codigo'] != codigo_actual:
            print(f"   ✨ Sugerencia: {recordado['codigo']} - {recordado['nombre']}")
            return jsonify({
                "es_correcta": False,
                "codigo_sugerido": recordado['codigo'],
                "nombre_sugerido": recordado['nombre'],
                "justificacion": f"📚 {recordado['tipo']}: {recordado['nombre']}",
                "confianza": recordado.get('score', 70) / 100
            })
        
        print(f"   ✓ Manteniendo cuenta actual")
        return jsonify({
            "es_correcta": True,
            "codigo_sugerido": codigo_actual,
            "nombre_sugerido": nombre_actual,
            "justificacion": "✓ Se mantiene cuenta actual",
            "confianza": 0.5
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e), "es_correcta": False}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        desc_f = data.get('descripcion', '').lower()
        codigo_correcto = data.get('codigo_correcto', '')
        
        nombre_real = obtener_nombre_real(codigo_correcto)
        if not nombre_real:
            return jsonify({"error": "Código no existe"}), 400
        
        FEEDBACK_DB[desc_f] = {'codigo': codigo_correcto, 'nombre': nombre_real}
        guardar_feedback(FEEDBACK_DB)
        
        patron = extraer_palabras_clave(desc_f)
        if patron:
            MEMORIA[patron] = {'codigo': codigo_correcto, 'nombre': nombre_real, 'score': 100, 'tipo': 'feedback'}
            guardar_memoria(MEMORIA)
        
        print(f"📝 Feedback: '{desc_f[:50]}' → {codigo_correcto}")
        return jsonify({"status": "✅ Aprendido"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memoria')
def ver_memoria():
    return jsonify({
        "memoria": len(MEMORIA),
        "feedback": len(FEEDBACK_DB),
        "ejemplos": dict(list(FEEDBACK_DB.items())[:20])
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "active",
        "memoria": len(MEMORIA),
        "feedback": len(FEEDBACK_DB),
        "historico": len(historico_desc_list)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

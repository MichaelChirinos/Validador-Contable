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

def cargar_memoria():
    if os.path.exists(MEMORIA_ARCHIVO):
        with open(MEMORIA_ARCHIVO, 'r', encoding='latin-1') as f:
            return json.load(f)
    return {}

def guardar_memoria(memoria):
    with open(MEMORIA_ARCHIVO, 'w', encoding='latin-1') as f:
        json.dump(memoria, f, indent=2)

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
    if not desc:
        return ""
    
    desc = desc.lower()
    # Eliminar códigos
    desc = re.sub(r'\b[a-z]{2,4}[\s-]*\d+\b', '', desc)
    desc = re.sub(r'dcc|fee|ee', '', desc, flags=re.IGNORECASE)
    desc = re.sub(r'\d+', '', desc)
    # Eliminar nombres de personas
    desc = re.sub(r'\b[a-z]+\s+[a-z]+\s+[a-z]+\b', '', desc)
    desc = re.sub(r'\b[a-z]+\s+[a-z]+\b', '', desc)
    # Limpiar
    desc = re.sub(r'[^\w\sáéíóúñ]', ' ', desc)
    
    palabras = re.findall(r'[a-záéíóúñ]{4,}', desc)
    return ' '.join(palabras[:3])

# --- CARGA DEL HISTÓRICO (optimizada) ---
print("📚 Cargando histórico...")
df_historico = pd.read_csv(
    'results.csv', 
    sep='\t',
    names=['Dscription', 'AcctName'],
    encoding='latin-1', 
    on_bad_lines='skip'
)
df_historico = df_historico.dropna(subset=['Dscription'])
df_historico['Dscription_lower'] = df_historico['Dscription'].str.lower()
historico_desc_list = df_historico['Dscription'].tolist()

# Crear un diccionario para búsqueda rápida
desc_to_acct = dict(zip(df_historico['Dscription_lower'], df_historico['AcctName']))

MEMORIA = cargar_memoria()
print(f"📚 Memoria: {len(MEMORIA)} patrones")
print(f"📊 Histórico: {len(historico_desc_list)} descripciones")

def recordar(descripcion):
    """Busca coincidencia de forma optimizada (sin iterrows)"""
    desc_lower = descripcion.lower()
    
    # 1. Buscar coincidencia EXACTA
    if desc_lower in desc_to_acct:
        nombre_historico = desc_to_acct[desc_lower]
        for code, name in catalogo_dict.items():
            if name == nombre_historico:
                return {'codigo': code, 'nombre': name, 'score': 100}
    
    # 2. Buscar en memoria por palabras clave
    patron = extraer_palabras_clave(descripcion)
    if patron and patron in MEMORIA:
        return MEMORIA[patron]
    
    # 3. Buscar por fuzzy matching (solo top 5, no todo el histórico)
    matches = process.extract(descripcion, historico_desc_list, limit=5, scorer=fuzz.token_set_ratio)
    
    for val, score in matches:
        if score > 75:
            nombre_historico = desc_to_acct.get(val.lower())
            if nombre_historico:
                for code, name in catalogo_dict.items():
                    if name == nombre_historico:
                        resultado = {'codigo': code, 'nombre': name, 'score': score}
                        # Aprender si es buena coincidencia
                        if score > 85 and patron:
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
        
        # Buscar en histórico (optimizado)
        recordado = recordar(desc_f)
        
        if recordado and recordado['codigo'] != codigo_actual:
            return jsonify({
                "es_correcta": False,
                "codigo_sugerido": recordado['codigo'],
                "nombre_sugerido": recordado['nombre'],
                "justificacion": f"📚 Histórico sugiere: {recordado['nombre']}",
                "confianza": recordado.get('score', 70) / 100
            })
        
        # Por defecto, mantener actual
        return jsonify({
            "es_correcta": True,
            "codigo_sugerido": codigo_actual,
            "nombre_sugerido": nombre_actual,
            "justificacion": "✓ Se mantiene cuenta actual",
            "confianza": 0.7
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e), "es_correcta": False}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        desc_f = data.get('descripcion', '')
        codigo_correcto = data.get('codigo_correcto', '')
        
        nombre_real = obtener_nombre_real(codigo_correcto)
        if not nombre_real:
            return jsonify({"error": "Código no existe"}), 400
        
        patron = extraer_palabras_clave(desc_f)
        if patron:
            MEMORIA[patron] = {'codigo': codigo_correcto, 'nombre': nombre_real, 'score': 100}
            guardar_memoria(MEMORIA)
        
        return jsonify({"status": "✅ Aprendido"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "active",
        "memoria": len(MEMORIA),
        "historico": len(historico_desc_list)
    })

@app.route('/')
def index():
    return "API de Auditoría Contable activa."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

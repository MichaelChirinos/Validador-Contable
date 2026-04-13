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
    """Extrae SOLO las palabras más importantes"""
    if not desc:
        return ""
    
    desc = desc.lower()
    # Eliminar códigos
    desc = re.sub(r'\b[a-z]{2,4}[\s-]*\d+\b', '', desc)
    desc = re.sub(r'dcc', '', desc)
    desc = re.sub(r'\d+', '', desc)
    # Eliminar nombres de personas (patrón: nombre apellido)
    desc = re.sub(r'\b[a-z]+\s+[a-z]+\s+[a-z]+\b', '', desc)
    desc = re.sub(r'\b[a-z]+\s+[a-z]+\b', '', desc)
    # Limpiar
    desc = re.sub(r'[^\w\s]', ' ', desc)
    
    palabras = re.findall(r'[a-záéíóúñ]+', desc)
    # Solo palabras significativas (longitud > 4)
    palabras = [p for p in palabras if len(p) > 4]
    return ' '.join(palabras[:3])

def recordar(descripcion, historico_df, historico_list):
    """Busca coincidencia EXACTA o muy similar"""
    patron = extraer_palabras_clave(descripcion)
    
    if not patron:
        return None
    
    # 1. Buscar en memoria
    if patron in MEMORIA:
        return MEMORIA[patron]
    
    # 2. Buscar coincidencia EXACTA en histórico
    for idx, row in historico_df.iterrows():
        if row['Dscription'].lower() == descripcion.lower():
            nombre_historico = row['AcctName']
            for code, name in catalogo_dict.items():
                if name == nombre_historico:
                    resultado = {'codigo': code, 'nombre': name, 'score': 100}
                    MEMORIA[patron] = resultado
                    guardar_memoria(MEMORIA)
                    return resultado
    
    # 3. Buscar coincidencia por palabras clave (más estricto)
    palabras_desc = set(patron.split())
    mejores = []
    
    for idx, row in historico_df.iterrows():
        desc_hist = row['Dscription'].lower()
        palabras_hist = set(extraer_palabras_clave(desc_hist).split())
        
        if palabras_hist:
            interseccion = palabras_desc.intersection(palabras_hist)
            if len(interseccion) >= 2:  # Al menos 2 palabras clave coinciden
                score = len(interseccion) / max(len(palabras_desc), len(palabras_hist)) * 100
                mejores.append((row, score))
    
    if mejores:
        mejores.sort(key=lambda x: x[1], reverse=True)
        mejor_row, score = mejores[0]
        if score > 60:
            nombre_historico = mejor_row['AcctName']
            for code, name in catalogo_dict.items():
                if name == nombre_historico:
                    resultado = {'codigo': code, 'nombre': name, 'score': score}
                    if score > 85:
                        MEMORIA[patron] = resultado
                        guardar_memoria(MEMORIA)
                    return resultado
    
    return None

# --- CARGA DEL HISTÓRICO ---
print("📚 Cargando histórico...")
df_historico = pd.read_csv(
    'results.csv', 
    sep='\t',  # Cambia a tabulación
    names=['Dscription', 'AcctName'],
    encoding='latin-1', 
    on_bad_lines='skip'
)
df_historico = df_historico.dropna(subset=['Dscription'])
historico_desc_list = df_historico['Dscription'].astype(str).tolist()

MEMORIA = cargar_memoria()
print(f"📚 Memoria: {len(MEMORIA)} patrones")
print(f"📊 Histórico: {len(historico_desc_list)} descripciones")

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        codigo_actual = str(cuenta_actual.get('AcctCode', ''))
        nombre_actual = cuenta_actual.get('AcctName', '')
        
        # Buscar en histórico
        recordado = recordar(desc_f, df_historico, historico_desc_list)
        
        if recordado and recordado['codigo'] != codigo_actual:
            return jsonify({
                "es_correcta": False,
                "codigo_sugerido": recordado['codigo'],
                "nombre_sugerido": recordado['nombre'],
                "justificacion": f"📚 Según histórico: {recordado['nombre']}",
                "confianza": recordado.get('score', 0) / 100
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

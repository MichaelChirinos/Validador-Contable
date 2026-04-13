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
    """Extrae palabras clave de forma genérica (sin hardcodeo)"""
    if not desc:
        return ""
    
    desc = desc.lower()
    
    # Eliminar códigos como DCC02563, F9Z-716, placa BZK-628
    desc = re.sub(r'\b[a-z]{2,4}[\s-]*\d{3,}\b', '', desc)
    # Eliminar números sueltos
    desc = re.sub(r'\b\d{3,}\b', '', desc)
    # Eliminar guiones y caracteres especiales
    desc = re.sub(r'[^\w\sáéíóúñ]', ' ', desc)
    
    # Extraer palabras (incluyendo acentos)
    palabras = re.findall(r'[a-záéíóúñ]+', desc)
    # Filtrar palabras muy cortas y comunes
    palabras_importantes = [p for p in palabras if len(p) > 3 and p not in ['para', 'por', 'con', 'sin', 'sobre']]
    
    # Limitar a 5 palabras clave
    return ' '.join(palabras_importantes[:5])

def recordar(descripcion, historico_df, historico_list):
    """Busca en memoria primero, luego en histórico"""
    patron = extraer_palabras_clave(descripcion)
    
    if not patron:
        return None
    
    # 1. Buscar en memoria
    if patron in MEMORIA:
        return MEMORIA[patron]
    
    # 2. Buscar en histórico (fuzzy)
    matches = process.extract(descripcion, historico_list, limit=5, scorer=fuzz.token_set_ratio)
    
    for val, score in matches:
        if score > 60:
            try:
                row = historico_df[historico_df['Dscription'] == val].iloc[0]
                nombre_historico = row['AcctName']
                # Buscar el código real en catálogo
                for code, name in catalogo_dict.items():
                    if name == nombre_historico:
                        # Aprender esta asociación
                        MEMORIA[patron] = {'codigo': code, 'nombre': name, 'veces': 1, 'score': score}
                        guardar_memoria(MEMORIA)
                        return {'codigo': code, 'nombre': name, 'score': score}
            except:
                continue
    
    return None

# --- CARGA DEL HISTÓRICO ---
print("📚 Cargando histórico...")
df_historico = pd.read_csv(
    'results.csv', 
    sep=';',
    names=['ItemCode', 'Dscription', 'AcctName', 'Proveedor', 
           'ItmsGrpNam', 'U_TipGasCos', 'U_TipOper', 'OcrCode3'], 
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
        
        # Buscar en memoria/histórico
        recordado = recordar(desc_f, df_historico, historico_desc_list)
        
        if recordado:
            codigo_sugerido = recordado['codigo']
            nombre_sugerido = recordado['nombre']
            es_correcta = (codigo_sugerido == codigo_actual)
            confianza = recordado.get('score', 70) / 100
            
            return jsonify({
                "es_correcta": es_correcta,
                "codigo_sugerido": codigo_sugerido,
                "nombre_sugerido": nombre_sugerido,
                "justificacion": f"📚 Coincidencia en histórico ({recordado.get('score', 0):.0f}%)",
                "confianza": confianza
            })
        
        # Sin coincidencia, mantener actual
        return jsonify({
            "es_correcta": True,
            "codigo_sugerido": codigo_actual,
            "nombre_sugerido": nombre_actual,
            "justificacion": "✓ Sin coincidencias, se mantiene cuenta actual",
            "confianza": 0.5
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
        MEMORIA[patron] = {'codigo': codigo_correcto, 'nombre': nombre_real, 'veces': 1}
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

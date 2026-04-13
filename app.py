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
catalogo_nombre_a_codigo = {v: k for k, v in catalogo_dict.items()}

def obtener_codigo_desde_nombre(nombre):
    """Convierte un nombre de cuenta a su código"""
    if not nombre:
        return None
    return catalogo_nombre_a_codigo.get(nombre)

def obtener_nombre_desde_codigo(codigo):
    return catalogo_dict.get(str(codigo))

def es_codigo_valido(codigo):
    """Verifica si el código es numérico de 4-6 dígitos"""
    if not codigo:
        return False
    codigo_str = str(codigo).strip()
    return bool(re.match(r'^\d{4,6}$', codigo_str))

# --- CARGA DEL HISTÓRICO ---
df_historico = pd.read_csv(
    'results.csv', 
    sep=';',
    names=['ItemCode', 'Dscription', 'AcctName', 'Proveedor', 
           'ItmsGrpNam', 'U_TipGasCos', 'U_TipOper', 'OcrCode3'],
    encoding='latin-1', 
    on_bad_lines='skip'
)
df_historico = df_historico.dropna(subset=['Dscription', 'AcctName'])
historico_desc_list = df_historico['Dscription'].tolist()
desc_to_acct = dict(zip(df_historico['Dscription'].str.lower(), df_historico['AcctName']))

MEMORIA = cargar_memoria()
print(f"📚 Memoria: {len(MEMORIA)} patrones")
print(f"📊 Histórico: {len(historico_desc_list)} registros")

def extraer_patron(desc):
    desc = desc.lower()
    desc = re.sub(r'dcc\s*\d+\s*-?\s*', '', desc)
    desc = re.sub(r'[a-z]+\s+[a-z]+\s+[a-z]+', '', desc)
    desc = re.sub(r'\d+', '#', desc)
    desc = re.sub(r'[^\w\s]', ' ', desc)
    palabras = re.findall(r'[a-záéíóúñ]{4,}', desc)
    return ' '.join(palabras[:3])

def buscar_contexto_historico(descripcion):
    matches = process.extract(descripcion, historico_desc_list, limit=3, scorer=fuzz.token_set_ratio)
    contexto = []
    for val, score in matches:
        if score > 50:
            nombre = desc_to_acct.get(val.lower(), '')
            if nombre:
                contexto.append(f"- '{val[:60]}' → {nombre}")
    return '\n'.join(contexto)

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        codigo_actual = str(cuenta_actual.get('AcctCode', ''))
        nombre_actual = cuenta_actual.get('AcctName', '')
        
        patron = extraer_patron(desc_f)
        
        # --- 1. Buscar en MEMORIA ---
        if patron in MEMORIA:
            recordado = MEMORIA[patron]
            if recordado['codigo'] != codigo_actual:
                return jsonify({
                    "es_correcta": False,
                    "codigo_sugerido": recordado['codigo'],
                    "nombre_sugerido": recordado['nombre'],
                    "justificacion": f"📚 Aprendido de {recordado['veces']} casos similares",
                    "confianza": 0.9
                })
            else:
                return jsonify({
                    "es_correcta": True,
                    "codigo_sugerido": codigo_actual,
                    "nombre_sugerido": nombre_actual,
                    "justificacion": "✓ Consistente con aprendizaje previo",
                    "confianza": 0.9
                })
        
        # --- 2. Usar IA ---
        contexto = buscar_contexto_historico(desc_f)
        
        prompt = f"""Eres un auditor contable. Analiza esta factura:

DESCRIPCIÓN: "{desc_f[:150]}"
CUENTA ACTUAL: {codigo_actual} - {nombre_actual}

CONTEXTO HISTÓRICO:
{contexto}

REGLAS:
- Si la cuenta actual es razonable, es_correcta = true
- Si no, sugiere la cuenta correcta (solo códigos numéricos de 4-6 dígitos)
- El código sugerido DEBE ser numérico, no texto

RESPONDE SOLO JSON:
{{"es_correcta": bool, "codigo_sugerido": "string (solo números)", "nombre_sugerido": "string", "justificacion": "string"}}"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300
        )
        
        resultado = json.loads(res.choices[0].message.content)
        
        # --- 3. CORREGIR RESPUESTAS INCORRECTAS ---
        codigo_sugerido = resultado.get('codigo_sugerido', '')
        nombre_sugerido = resultado.get('nombre_sugerido', '')
        
        # Si el código sugerido es texto (ej: "SEGUROS DE PERSONAL"), convertirlo
        if not es_codigo_valido(codigo_sugerido) and nombre_sugerido:
            codigo_correcto = obtener_codigo_desde_nombre(nombre_sugerido)
            if codigo_correcto:
                codigo_sugerido = codigo_correcto
                resultado['codigo_sugerido'] = codigo_correcto
        
        # Si no hay código sugerido, usar el actual
        if not es_codigo_valido(codigo_sugerido):
            codigo_sugerido = codigo_actual
            nombre_sugerido = nombre_actual
            resultado['codigo_sugerido'] = codigo_actual
            resultado['nombre_sugerido'] = nombre_actual
        
        # Si el código sugerido es igual al actual, debe ser true
        if codigo_sugerido == codigo_actual:
            resultado['es_correcta'] = True
        
        # --- 4. APRENDER ---
        if patron:
            if patron not in MEMORIA:
                MEMORIA[patron] = {
                    'codigo': codigo_sugerido,
                    'nombre': nombre_sugerido,
                    'veces': 1
                }
            else:
                MEMORIA[patron]['veces'] += 1
            guardar_memoria(MEMORIA)
        
        return jsonify(resultado)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e), "es_correcta": False, "codigo_sugerido": codigo_actual, "nombre_sugerido": nombre_actual}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        desc_f = data.get('descripcion', '')
        codigo_correcto = data.get('codigo_correcto', '')
        
        nombre_real = obtener_nombre_desde_codigo(codigo_correcto)
        if not nombre_real:
            return jsonify({"error": "Código no existe"}), 400
        
        patron = extraer_patron(desc_f)
        MEMORIA[patron] = {
            'codigo': codigo_correcto,
            'nombre': nombre_real,
            'veces': MEMORIA.get(patron, {}).get('veces', 0) + 1
        }
        guardar_memoria(MEMORIA)
        
        return jsonify({"status": "✅ Aprendido"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memoria')
def ver_memoria():
    return jsonify({
        "total_patrones": len(MEMORIA),
        "patrones": dict(list(MEMORIA.items())[:30])
    })

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

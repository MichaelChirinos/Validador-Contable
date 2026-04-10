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

# --- MEMORIA Y MÉTRICAS ---
MEMORIA_ARCHIVO = 'memoria.json'
MEMORIA = {}

def cargar_memoria():
    global MEMORIA
    if os.path.exists(MEMORIA_ARCHIVO):
        with open(MEMORIA_ARCHIVO, 'r', encoding='latin-1') as f:
            MEMORIA = json.load(f)
    print(f"📚 Memoria: {len(MEMORIA)} patrones")

def guardar_memoria():
    with open(MEMORIA_ARCHIVO, 'w', encoding='latin-1') as f:
        json.dump(MEMORIA, f, indent=2)

def extraer_patron(desc):
    desc = desc.lower()
    desc = re.sub(r'\d+', '#', desc)
    desc = re.sub(r'\d*\s*(kgs?|kg|saco?s?|unidad|unid|gal|lt|l)', '', desc)
    desc = re.sub(r'\([^)]+\)', '', desc)
    return ' '.join(desc.split())[:80]

def recordar(desc):
    patron = extraer_patron(desc)
    if patron in MEMORIA:
        return MEMORIA[patron]
    mejores = process.extract(patron, list(MEMORIA.keys()), limit=1, scorer=fuzz.token_set_ratio)
    if mejores and mejores[0][1] > 85:
        return MEMORIA[mejores[0][0]]
    return None

def aprender(desc, codigo, nombre, correcta):
    patron = extraer_patron(desc)
    if patron not in MEMORIA:
        MEMORIA[patron] = {'codigo': codigo, 'nombre': nombre, 'veces': 1, 'aciertos': 1 if correcta else 0}
    else:
        MEMORIA[patron]['veces'] += 1
        if correcta:
            MEMORIA[patron]['aciertos'] += 1
    guardar_memoria()

# --- CARGA INICIAL ---
cargar_memoria()

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

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        codigo_actual = str(cuenta_actual.get('AcctCode', ''))
        nombre_actual = cuenta_actual.get('AcctName', '')
        
        # --- 1. Buscar en memoria ---
        recordado = recordar(desc_f)
        if recordado:
            es_correcta = (recordado['codigo'] == codigo_actual)
            return jsonify({
                "es_correcta": es_correcta,
                "codigo_sugerido": recordado['codigo'] if not es_correcta else codigo_actual,
                "nombre_sugerido": recordado['nombre'] if not es_correcta else nombre_actual,
                "justificacion": f"📚 Memoria ({recordado['veces']} casos)",
                "confianza": recordado['aciertos'] / recordado['veces']
            })
        
        # --- 2. Contexto histórico ---
        m_h = process.extract(desc_f, historico_desc_list, limit=2, scorer=fuzz.token_set_ratio)
        ctx = ""
        for val, score in m_h:
            if score > 40:
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                ctx += f"- {val[:50]} → {row['AcctName']}\n"
        
        # --- 3. IA ---
        prompt = f"""Valida: "{desc_f[:80]}"
Actual: {codigo_actual} - {nombre_actual}
Ref: {ctx}
JSON: {{"es_correcta":bool, "codigo_sugerido":"str", "nombre_sugerido":"str", "justificacion":"str"}}"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200
        )
        
        resultado = json.loads(res.choices[0].message.content)
        codigo_sugerido = str(resultado.get('codigo_sugerido', ''))
        
        # --- 4. FIX CRÍTICO: Si el código coincide, es correcta ---
        if codigo_sugerido == codigo_actual:
            resultado['es_correcta'] = True
            resultado['justificacion'] = f"✓ {resultado.get('justificacion', 'Coincidencia de código')}"
        
        # --- 5. Aprender ---
        aprender(desc_f, resultado.get('codigo_sugerido', codigo_actual), 
                resultado.get('nombre_sugerido', nombre_actual), 
                resultado.get('es_correcta', False))
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e), "es_correcta": False}), 500

@app.route('/health')
def health():
    return jsonify({"status": "active", "memoria": len(MEMORIA)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

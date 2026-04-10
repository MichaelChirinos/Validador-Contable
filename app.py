import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process, fuzz
from groq import Groq
import json
import re

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CARGA DE CATÁLOGO (para validar códigos) ---
df_catalogo = pd.read_csv(
    'results_1.csv', 
    sep=';',
    names=['AcctCode', 'AcctName'],
    encoding='latin-1',
    on_bad_lines='skip'
)
catalogo_codigos = df_catalogo['AcctCode'].astype(str).tolist()

def es_codigo_valido(codigo):
    """Código de cuenta debe ser 4-6 dígitos numéricos"""
    if not codigo:
        return False
    return bool(re.match(r'^\d{4,6}$', str(codigo).strip()))

def corregir_codigo_sugerido(codigo_sugerido, codigo_actual):
    """Corrige códigos alucinados"""
    if es_codigo_valido(codigo_sugerido):
        return codigo_sugerido
    
    # Buscar en catálogo por similitud
    if catalogo_codigos:
        match = process.extractOne(str(codigo_sugerido), catalogo_codigos, scorer=fuzz.token_set_ratio)
        if match and match[1] > 60:
            return match[0]
    
    # Si no hay match, usar el código actual
    return codigo_actual

# --- (resto de tu código: cargar histórico, memoria, etc.) ---

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        codigo_actual = str(cuenta_actual.get('AcctCode', ''))
        nombre_actual = cuenta_actual.get('AcctName', '')
        
        # ... (código de fuzzy matching y llamada a IA) ...
        
        resultado = json.loads(res.choices[0].message.content)
        
        # --- VALIDAR CÓDIGO SUGERIDO ---
        codigo_sugerido_raw = str(resultado.get('codigo_sugerido', codigo_actual))
        codigo_sugerido_corregido = corregir_codigo_sugerido(codigo_sugerido_raw, codigo_actual)
        
        # Si el código fue corregido, actualizar
        if codigo_sugerido_corregido != codigo_sugerido_raw:
            resultado['codigo_sugerido'] = codigo_sugerido_corregido
            resultado['justificacion'] = f"🔧 Corregido: '{codigo_sugerido_raw}' no es válido. {resultado.get('justificacion', '')}"
        
        # Si el código coincide con el actual, es correcta
        if str(resultado.get('codigo_sugerido', '')) == codigo_actual:
            resultado['es_correcta'] = True
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e), "es_correcta": False}), 500

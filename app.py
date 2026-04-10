import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process, fuzz
from groq import Groq
import json
import re

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- MEMORIA DE APRENDIZAJE (la clave de la consistencia) ---
MEMORIA = {}

def extraer_patron(desc):
    """Extrae patrón general de la descripción (ej: 'producto químico en saco')"""
    desc = desc.lower()
    # Eliminar números específicos
    desc = re.sub(r'\d+', '#', desc)
    # Eliminar medidas (KGS, KG, SAC, etc.)
    desc = re.sub(r'\d*\s*(kgs?|kg|saco?s?|unidad|unid|gal)', '', desc)
    # Eliminar paréntesis y su contenido
    desc = re.sub(r'\([^)]+\)', '', desc)
    # Limpiar espacios
    desc = ' '.join(desc.split())
    return desc[:80]  # Limitar longitud

def recordar(descripcion):
    """Buscar si ya aprendimos algo similar"""
    patron = extraer_patron(descripcion)
    
    # Coincidencia exacta
    if patron in MEMORIA:
        return MEMORIA[patron]
    
    # Coincidencia fuzzy
    mejores = process.extract(patron, list(MEMORIA.keys()), limit=1, scorer=fuzz.token_set_ratio)
    if mejores and mejores[0][1] > 85:
        return MEMORIA[mejores[0][0]]
    
    return None

def aprender(descripcion, codigo, nombre, correcta):
    """Guardar patrón para futuras consultas"""
    patron = extraer_patron(descripcion)
    
    if patron not in MEMORIA:
        MEMORIA[patron] = {
            'codigo': codigo,
            'nombre': nombre,
            'veces': 1,
            'correcta': correcta
        }
    else:
        MEMORIA[patron]['veces'] += 1

# --- CARGA INICIAL ---
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
        
        # --- 1. Buscar en MEMORIA (consistencia garantizada) ---
        recordado = recordar(desc_f)
        
        if recordado:
            # Ya aprendimos este patrón
            if recordado['codigo'] == codigo_actual:
                return jsonify({
                    "es_correcta": True,
                    "codigo_sugerido": codigo_actual,
                    "nombre_sugerido": nombre_actual,
                    "justificacion": "✓ Consistente con aprendizaje previo"
                })
            else:
                return jsonify({
                    "es_correcta": False,
                    "codigo_sugerido": recordado['codigo'],
                    "nombre_sugerido": recordado['nombre'],
                    "justificacion": f"📚 Según patrón aprendido: {recordado['codigo']} - {recordado['nombre']}"
                })
        
        # --- 2. Si no hay memoria, buscar contexto ---
        m_h = process.extract(desc_f, historico_desc_list, limit=2, scorer=fuzz.token_set_ratio)
        ctx = ""
        for val, score in m_h:
            if score > 40:
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                ctx += f"- {val[:50]} → {row['AcctName']}\n"
        
        # --- 3. Prompt MINIMALISTA (sin ejemplos fijos) ---
        prompt = f"""Valida: "{desc_f[:80]}"
Actual: {codigo_actual} - {nombre_actual}
Ref: {ctx}
Reglas: Sé consistente. Si dudas, mantén actual.
JSON: {{"es_correcta":bool, "codigo_sugerido":"str", "nombre_sugerido":"str", "justificacion":"str"}}"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200
        )
        
        resultado = json.loads(res.choices[0].message.content)
        
        # --- 4. APRENDER para futuras consultas ---
        aprender(desc_f, resultado.get('codigo_sugerido', codigo_actual), 
                resultado.get('nombre_sugerido', nombre_actual), 
                resultado.get('es_correcta', False))
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e), "es_correcta": False}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

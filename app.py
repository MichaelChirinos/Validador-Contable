import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process, fuzz
from groq import Groq
import json
import re

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CARGA DE DATOS OPTIMIZADA ---
def cargar_datos():
    # Solo cargamos las columnas necesarias para ahorrar RAM y agilizar búsqueda
    df_h = pd.read_csv('results.csv', sep=';', names=['ItemCode', 'Dscription', 'AcctName'], 
                       usecols=['Dscription', 'AcctName'], encoding='latin-1', on_bad_lines='skip').dropna()
    df_c = pd.read_csv('results_1.csv', sep=';', names=['AcctCode', 'AcctName'], 
                       encoding='latin-1', on_bad_lines='skip')
    return df_h, df_c

df_historico, df_catalogo = cargar_datos()

def limpiar_ultra_light(desc):
    """Limpieza agresiva para gastar menos tokens en el prompt"""
    d = desc.lower()
    d = re.sub(r'^(dcc\s+)?([a-z]+\s+){1,2}//\s+', '', d) # Quita nombres iniciales
    d = re.sub(r'\d+', '', d) # Quita números (años, series) para generalizar
    return " ".join(d.split())[:80]

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_original = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        
        desc_limpia = limpiar_ultra_light(desc_original)
        
        # 1. HISTORIAL (Busca en las 42k filas localmente)
        matches_h = process.extract(desc_limpia, df_historico['Dscription'].tolist(), limit=2, scorer=fuzz.token_set_ratio)
        ctx = ""
        for val, score in matches_h:
            if score > 70:
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                ctx += f"H:{val[:40]}->{row['AcctName']}|" # Formato ultra compacto

        # 2. CATÁLOGO (Solo 5 opciones)
        matches_c = process.extract(desc_limpia, df_catalogo['AcctName'].tolist(), limit=5, scorer=fuzz.token_set_ratio)
        catalogo = [{"c": df_catalogo[df_catalogo['AcctName'] == m[0]].iloc[0]['AcctCode'], "n": m[0]} for m in matches_c]

        # 3. PROMPT COMPRIMIDO (Ahorro de ~40% de tokens)
        prompt = f"""Audit Manuchar. Desc: "{desc_original}"
        Current: {cuenta_actual.get('AcctCode')} - {cuenta_actual.get('AcctName')}
        History: {ctx}
        Catalog: {catalogo}
        Task: If current is wrong/generic, suggest best from Catalog. Focus on Noun (Object vs Service).
        Output JSON: {{"es_correcta":bool, "codigo_sugerido":"str", "nombre_sugerido":"str", "justificacion":"str"}}"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=150 # Limita la respuesta para no gastar de más
        )
        
        resultado = json.loads(res.choices[0].message.content)

        # Validación post-IA (Sigue siendo ML-like)
        if not resultado['es_correcta']:
            check = process.extractOne(resultado['nombre_sugerido'], df_catalogo['AcctName'].tolist())
            if check and check[1] > 85:
                row_c = df_catalogo[df_catalogo['AcctName'] == check[0]].iloc[0]
                resultado.update({"codigo_sugerido": row_c['AcctCode'], "nombre_sugerido": row_c['AcctName']})

        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 

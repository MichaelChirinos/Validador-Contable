import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process, fuzz
from groq import Groq
import json

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CARGA LOCAL ---
def cargar_archivos():
    columnas_historico = [
        'ItemCode', 'Dscription', 'AcctName', 'Proveedor', 
        'ItmsGrpNam', 'U_TipGasCos', 'U_TipOper', 'OcrCode3'
    ]
    
    df_h = pd.read_csv(
        'results.csv', 
        sep=';',
        names=columnas_historico, 
        encoding='latin-1', 
        on_bad_lines='skip'
    )
    df_h = df_h.dropna(subset=['Dscription'])
    df_h = df_h.replace(['nan', 'NULL'], pd.NA)
    
    columnas_catalogo = ['AcctCode', 'AcctName']
    
    df_c = pd.read_csv(
        'results_1.csv', 
        sep=';',
        names=columnas_catalogo,
        encoding='latin-1',
        on_bad_lines='skip'
    )
    
    df_c = df_c.dropna(subset=['AcctCode', 'AcctName'])
    df_c = df_c.replace(['nan', 'NULL'], pd.NA)
    df_c = df_c[df_c['AcctCode'].astype(str).str.len() >= 4]
    
    print(f"✅ Histórico: {len(df_h)} filas")
    print(f"✅ Catálogo: {len(df_c)} cuentas")
    
    return df_h, df_c

df_historico, df_catalogo = cargar_archivos()
historico_desc_list = df_historico['Dscription'].astype(str).tolist()
catalogo_nombres = df_catalogo['AcctName'].astype(str).tolist()

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        codigo_actual = str(cuenta_actual.get('AcctCode', ''))
        nombre_actual = cuenta_actual.get('AcctName', '')
        
        # Contexto histórico (solo referencia)
        m_h = process.extract(desc_f, historico_desc_list, limit=3, scorer=fuzz.token_set_ratio)
        
        ctx_h = []
        for val, score in m_h:
            if score > 40:
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                ctx_h.append(f"- '{val[:50]}' → {row['AcctName']}")
        
        ctx_str = "\n".join(ctx_h) if ctx_h else "Sin coincidencias en histórico"
        
        # Prompt permisivo con capacidad de sugerir
        prompt = f"""Eres auditor contable senior. Valida o corrige esta factura.

DESCRIPCIÓN: "{desc_f[:100]}"
CUENTA ACTUAL: {codigo_actual} - {nombre_actual}

REFERENCIA (solo guía, no obligación):
{ctx_str}

REGLAS:
- Si la cuenta actual es razonable → true
- Si NO es razonable → false y SUGIERE la cuenta correcta (usa tu conocimiento contable)
- Materiales/insumos → cuentas de compras (631xxx)
- Servicios → cuentas de gastos (631xxx, 953xxx)
- Factura de reserva es válida para insumos si ese es el criterio de la empresa

RESPONDE SOLO JSON:
{{"es_correcta": bool, "codigo_sugerido": "string", "nombre_sugerido": "string", "justificacion": "string"}}"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,  # Un poco más de creatividad para sugerir
            max_tokens=300
        )
        
        resultado = json.loads(res.choices[0].message.content)
        
        # Asegurar que el código sugerido tenga al menos 4 dígitos
        if resultado.get('codigo_sugerido') and len(str(resultado['codigo_sugerido'])) < 4:
            resultado['codigo_sugerido'] = codigo_actual
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e), "es_correcta": False}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "active",
        "historico": len(df_historico),
        "catalogo": len(df_catalogo)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

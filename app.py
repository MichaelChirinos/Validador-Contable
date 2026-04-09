import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process, fuzz
from groq import Groq

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CARGA LOCAL ---
def cargar_archivos():
    # 1. Histórico (CSV sin cabeceras, 8 columnas)
    columnas_historico = [
        'ItemCode', 'Dscription', 'AcctName', 'Proveedor', 
        'ItmsGrpNam', 'U_TipGasCos', 'U_TipOper', 'OcrCode3'
    ]
    
    df_h = pd.read_csv(
        'results.csv', 
        sep=';',
        names=columnas_historico,  # ← Asignamos nombres manualmente
        encoding='latin-1', 
        on_bad_lines='skip'
    )
    df_h = df_h.dropna(subset=['Dscription'])
    df_h = df_h.replace(['nan', 'NULL'], pd.NA)
    
    # 2. Catálogo SAP (CSV sin cabeceras, 2 columnas: código y nombre)
    # Asumo que la primera columna es AcctCode y la segunda AcctName
    columnas_catalogo = ['AcctCode', 'AcctName']
    
    df_c = pd.read_csv(
        'catalogo.csv', 
        sep=';',
        names=columnas_catalogo,  # ← Asignamos nombres manualmente
        encoding='latin-1',
        on_bad_lines='skip'
    )
    
    # Limpiar catálogo
    df_c = df_c.dropna(subset=['AcctCode', 'AcctName'])
    df_c = df_c.replace(['nan', 'NULL'], pd.NA)
    
    # Filtrar cuentas de registro (4-6 dígitos)
    df_c = df_c[df_c['AcctCode'].astype(str).str.len() >= 4]
    
    print(f"✅ Histórico: {len(df_h)} filas")
    print(f"✅ Catálogo: {len(df_c)} cuentas")
    print(f"📊 Ejemplo catálogo:\n{df_c.head(3)}")
    
    return df_h, df_c

df_historico, df_catalogo = cargar_archivos()
historico_desc_list = df_historico['Dscription'].astype(str).tolist()
catalogo_nombres = df_catalogo['AcctName'].astype(str).tolist()

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')[:150]
        cuenta_f = data.get('cuenta_actual', {})

        # Fuzzy matching en histórico
        m_h = process.extract(desc_f, historico_desc_list, limit=3, scorer=fuzz.token_set_ratio)
        
        ctx_h = []
        for val, score in m_h:
            if score > 50:
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                ctx_h.append(f"'{val[:40]}'→{row['AcctName']}")
        
        ctx_str = " | ".join(ctx_h) if ctx_h else "Sin coincidencias"
        
        # Fuzzy matching en catálogo
        m_c = process.extract(desc_f, catalogo_nombres, limit=5, scorer=fuzz.token_set_ratio)
        
        opciones_red = []
        for nombre, score in m_c:
            row = df_catalogo[df_catalogo['AcctName'] == nombre].iloc[0]
            opciones_red.append(f"{row['AcctCode']}-{nombre[:35]}")
        
        opciones_str = " | ".join(opciones_red) if opciones_red else "Sin opciones"
        
        # Prompt minimalista
        prompt = f"""Auditor SAP. Valida: "{desc_f[:80]}"
Actual: {cuenta_f.get('AcctCode')} - {cuenta_f.get('AcctName')}
Histórico: {ctx_str}
Opciones: {opciones_str}
Reglas: Prioriza histórico. Si coincide actual→true.
JSON: {{"es_correcta":bool, "codigo_sugerido":"str", "nombre_sugerido":"str", "justificacion":"str"}}"""
        
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300
        )
        
        return res.choices[0].message.content, 200, {'Content-Type': 'application/json'}
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

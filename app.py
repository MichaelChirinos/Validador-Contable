import os
import pandas as pd
from flask import Flask, request, jsonify
from thefuzz import process, fuzz
from groq import Groq

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- CARGA LOCAL ---
def cargar_archivos():
    # Histórico (42k registros)
    df_h = pd.read_csv(
        'results.csv', 
        sep=';', 
        names=['ItemCode', 'Dscription', 'AcctName', 'Proveedor', 
               'ItmsGrpNam', 'U_TipGasCos', 'U_TipOper', 'OcrCode3'], 
        encoding='latin-1', 
        on_bad_lines='skip'
    )
    
    # Catálogo SAP
    df_c = pd.read_csv('catalogo.csv', sep=';', encoding='latin-1')
    df_c = df_c[df_c['AcctCode'].astype(str).str.len() >= 4]
    
    # Limpieza
    df_h = df_h.dropna(subset=['Dscription'])
    df_h = df_h.replace(['nan', 'NULL'], pd.NA)
    
    # Optimización: crear lista una sola vez
    df_h['desc_lower'] = df_h['Dscription'].astype(str).str.lower()
    
    return df_h, df_c

df_historico, df_catalogo = cargar_archivos()
historico_desc_list = df_historico['Dscription'].astype(str).tolist()
catalogo_nombres = df_catalogo['AcctName'].astype(str).tolist()

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')[:150]  # Limitar longitud
        cuenta_f = data.get('cuenta_actual', {})
        
        # --- FUZZY MATCHING LOCAL ---
        # Búsqueda en histórico (top 3)
        m_h = process.extract(
            desc_f, 
            historico_desc_list, 
            limit=3, 
            scorer=fuzz.token_set_ratio
        )
        
        # Contexto compacto
        ctx_h = []
        for val, score in m_h:
            if score > 50:  # Solo si es relevante
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                ctx_h.append(f"'{val[:40]}'→{row['AcctName']}")
        
        ctx_str = " | ".join(ctx_h) if ctx_h else "Sin coincidencias"
        
        # Filtrado de catálogo (top 5)
        m_c = process.extract(
            desc_f, 
            catalogo_nombres, 
            limit=5, 
            scorer=fuzz.token_set_ratio
        )
        
        # Opciones en formato compacto
        opciones_red = []
        for nombre, score in m_c:
            row = df_catalogo[df_catalogo['AcctName'] == nombre].iloc[0]
            opciones_red.append(f"{row['AcctCode']}-{nombre[:35]}")
        
        opciones_str = " | ".join(opciones_red)
        
        # --- PROMPT MINIMALISTA ---
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
            max_tokens=300  # Limitar output también
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

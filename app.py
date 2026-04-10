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

# --- MEMORIA PERSISTENTE (aprende para siempre) ---
MEMORIA_ARCHIVO = 'memoria.json'
METRICAS_ARCHIVO = 'metricas.json'

def cargar_memoria():
    if os.path.exists(MEMORIA_ARCHIVO):
        with open(MEMORIA_ARCHIVO, 'r', encoding='latin-1') as f:
            return json.load(f)
    return {}

def guardar_memoria(memoria):
    with open(MEMORIA_ARCHIVO, 'w', encoding='latin-1') as f:
        json.dump(memoria, f, indent=2)

def cargar_metricas():
    if os.path.exists(METRICAS_ARCHIVO):
        with open(METRICAS_ARCHIVO, 'r', encoding='latin-1') as f:
            return json.load(f)
    return {'total': 0, 'acertados': 0, 'historial': []}

def guardar_metricas(metricas):
    with open(METRICAS_ARCHIVO, 'w', encoding='latin-1') as f:
        json.dump(metricas, f, indent=2)

def extraer_patron(desc):
    """Extrae patrón general de la descripción"""
    desc = desc.lower()
    desc = re.sub(r'\d+', '#', desc)
    desc = re.sub(r'\d*\s*(kgs?|kg|saco?s?|unidad|unid|gal|lt|l)', '', desc)
    desc = re.sub(r'\([^)]+\)', '', desc)
    desc = ' '.join(desc.split())
    return desc[:80]

def recordar(descripcion):
    """Buscar si ya aprendimos algo similar"""
    patron = extraer_patron(descripcion)
    
    if patron in MEMORIA:
        return MEMORIA[patron]
    
    # Búsqueda fuzzy en memoria
    mejores = process.extract(patron, list(MEMORIA.keys()), limit=1, scorer=fuzz.token_set_ratio)
    if mejores and mejores[0][1] > 85:
        return MEMORIA[mejores[0][0]]
    
    return None

def aprender(descripcion, codigo, nombre, correcta, feedback_manual=False):
    """Guardar patrón para futuras consultas"""
    patron = extraer_patron(descripcion)
    
    if patron not in MEMORIA:
        MEMORIA[patron] = {
            'codigo': codigo,
            'nombre': nombre,
            'veces': 1,
            'aciertos': 1 if correcta else 0,
            'ultimo': datetime.now().isoformat(),
            'feedback_manual': feedback_manual
        }
    else:
        MEMORIA[patron]['veces'] += 1
        if correcta:
            MEMORIA[patron]['aciertos'] += 1
        MEMORIA[patron]['ultimo'] = datetime.now().isoformat()
    
    guardar_memoria(MEMORIA)

def registrar_metrica(descripcion, fue_correcta, codigo_usado, codigo_sugerido):
    """Registrar métricas para tracking de precisión"""
    metricas = cargar_metricas()
    metricas['total'] += 1
    if fue_correcta:
        metricas['acertados'] += 1
    
    metricas['historial'].append({
        'fecha': datetime.now().isoformat(),
        'descripcion': descripcion[:100],
        'codigo_usado': codigo_usado,
        'codigo_sugerido': codigo_sugerido,
        'correcta': fue_correcta
    })
    
    # Mantener solo últimos 1000 registros
    if len(metricas['historial']) > 1000:
        metricas['historial'] = metricas['historial'][-1000:]
    
    guardar_metricas(metricas)
    return metricas

# --- CARGA INICIAL ---
MEMORIA = cargar_memoria()
print(f"📚 Memoria cargada: {len(MEMORIA)} patrones aprendidos")

metricas = cargar_metricas()
precision = (metricas['acertados'] / metricas['total'] * 100) if metricas['total'] > 0 else 0
print(f"📊 Precisión actual: {precision:.1f}% ({metricas['acertados']}/{metricas['total']})")

# Cargar histórico
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
        
        # --- 1. Buscar en MEMORIA ---
        recordado = recordar(desc_f)
        
        if recordado:
            # Ya aprendimos este patrón
            es_correcta = (recordado['codigo'] == codigo_actual)
            resultado = {
                "es_correcta": es_correcta,
                "codigo_sugerido": recordado['codigo'] if not es_correcta else codigo_actual,
                "nombre_sugerido": recordado['nombre'] if not es_correcta else nombre_actual,
                "justificacion": f"📚 Basado en {recordado['veces']} casos previos (precisión: {recordado['aciertos']/recordado['veces']*100:.0f}%)",
                "confianza": recordado['aciertos'] / recordado['veces']
            }
            
            registrar_metrica(desc_f, es_correcta, codigo_actual, resultado['codigo_sugerido'])
            return jsonify(resultado)
        
        # --- 2. Buscar contexto histórico ---
        m_h = process.extract(desc_f, historico_desc_list, limit=2, scorer=fuzz.token_set_ratio)
        ctx = ""
        for val, score in m_h:
            if score > 40:
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                ctx += f"- {val[:50]} → {row['AcctName']}\n"
        
        # --- 3. Llamar a IA (solo cuando no hay memoria) ---
        prompt = f"""Valida: "{desc_f[:80]}"
Actual: {codigo_actual} - {nombre_actual}
Ref: {ctx}
Sé consistente con casos similares.
JSON: {{"es_correcta":bool, "codigo_sugerido":"str", "nombre_sugerido":"str", "justificacion":"str"}}"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200
        )
        
        resultado = json.loads(res.choices[0].message.content)
        
        # --- 4. Aprender ---
        es_correcta = resultado.get('es_correcta', False)
        aprender(desc_f, resultado.get('codigo_sugerido', codigo_actual), 
                resultado.get('nombre_sugerido', nombre_actual), es_correcta)
        
        registrar_metrica(desc_f, es_correcta, codigo_actual, resultado.get('codigo_sugerido', codigo_actual))
        
        # Agregar confianza
        resultado['confianza'] = 0.5  # Primera vez, confianza media
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e), "es_correcta": False}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint para corregir manualmente y mejorar precisión"""
    data = request.json
    desc_f = data.get('descripcion')
    codigo_correcto = data.get('codigo_correcto')
    nombre_correcto = data.get('nombre_correcto')
    
    # Forzar aprendizaje con feedback manual
    aprender(desc_f, codigo_correcto, nombre_correcto, True, feedback_manual=True)
    
    return jsonify({"status": "✅ Aprendizaje reforzado"})

@app.route('/metricas')
def obtener_metricas():
    """Ver precisión actual"""
    metricas = cargar_metricas()
    precision = (metricas['acertados'] / metricas['total'] * 100) if metricas['total'] > 0 else 0
    
    return jsonify({
        "total_evaluaciones": metricas['total'],
        "acertados": metricas['acertados'],
        "precisión_porcentaje": round(precision, 2),
        "patrones_aprendidos": len(MEMORIA),
        "ultimos_10": metricas['historial'][-10:]
    })

@app.route('/health')
def health():
    metricas = cargar_metricas()
    precision = (metricas['acertados'] / metricas['total'] * 100) if metricas['total'] > 0 else 0
    return jsonify({
        "status": "active",
        "patrones_aprendidos": len(MEMORIA),
        "precision_actual": f"{precision:.1f}%",
        "total_evaluaciones": metricas['total']
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

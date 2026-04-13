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
catalogo_codigos = df_catalogo['AcctCode'].astype(str).tolist()
catalogo_dict = dict(zip(df_catalogo['AcctCode'].astype(str), df_catalogo['AcctName']))

def obtener_nombre_real(codigo):
    return catalogo_dict.get(str(codigo), None)

def extraer_patron(desc):
    """Extrae patrón general de la descripción"""
    desc = desc.lower()
    desc = re.sub(r'\d+', '#', desc)
    desc = re.sub(r'\d*\s*(kgs?|kg|saco?s?|unidad|unid|gal|lt|l)', '', desc)
    desc = re.sub(r'\([^)]+\)', '', desc)
    desc = re.sub(r'[^\w\s]', ' ', desc)  # Eliminar puntuación
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

def aprender(descripcion, codigo, nombre):
    """Guardar patrón para futuras consultas (solo si es confiable)"""
    patron = extraer_patron(descripcion)
    
    # Evitar aprender patrones muy cortos o genéricos
    if len(patron) < 10:
        return
    
    if patron not in MEMORIA:
        MEMORIA[patron] = {
            'codigo': codigo,
            'nombre': nombre,
            'veces': 1
        }
    else:
        # Si ya existe, incrementar contador
        MEMORIA[patron]['veces'] += 1
    
    guardar_memoria(MEMORIA)

# --- CARGA DEL HISTÓRICO (42k filas) y APRENDIZAJE INICIAL ---
print("📚 Cargando histórico de 42k filas...")
df_historico = pd.read_csv(
    'results.csv', 
    sep=';',
    names=['ItemCode', 'Dscription', 'AcctName', 'Proveedor', 
           'ItmsGrpNam', 'U_TipGasCos', 'U_TipOper', 'OcrCode3'], 
    encoding='latin-1', 
    on_bad_lines='skip'
)
df_historico = df_historico.dropna(subset=['Dscription'])
df_historico = df_historico.drop_duplicates(subset=['Dscription'])  # Evitar duplicados
historico_desc_list = df_historico['Dscription'].astype(str).tolist()

# --- APRENDER DEL HISTÓRICO COMPLETO (solo una vez) ---
print("🧠 Aprendiendo del histórico...")
MEMORIA = cargar_memoria()

# Si la memoria está vacía, aprender del histórico
if len(MEMORIA) == 0:
    print("   Memoria vacía. Precargando desde histórico...")
    count = 0
    for idx, row in df_historico.iterrows():
        desc = row['Dscription']
        acct_name = row['AcctName']
        # Buscar el código real en catálogo
        codigo_real = None
        for code, name in catalogo_dict.items():
            if name == acct_name:
                codigo_real = code
                break
        
        if codigo_real and len(desc) > 10:
            aprender(desc, codigo_real, acct_name)
            count += 1
            if count % 5000 == 0:
                print(f"   Aprendidas {count} descripciones...")
    
    print(f"✅ Memoria precargada con {len(MEMORIA)} patrones únicos")
else:
    print(f"✅ Memoria ya existente: {len(MEMORIA)} patrones")

print(f"📊 Histórico listo: {len(historico_desc_list)} descripciones")

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_f = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        codigo_actual = str(cuenta_actual.get('AcctCode', ''))
        nombre_actual = cuenta_actual.get('AcctName', '')
        
        # --- 1. Buscar en MEMORIA (aprendizaje previo del histórico) ---
        recordado = recordar(desc_f)
        
        if recordado:
            codigo_sugerido = recordado['codigo']
            nombre_sugerido = recordado['nombre']
            es_correcta = (codigo_sugerido == codigo_actual)
            confianza = min(1.0, recordado['veces'] / 10)
            
            return jsonify({
                "es_correcta": es_correcta,
                "codigo_sugerido": codigo_sugerido,
                "nombre_sugerido": nombre_sugerido,
                "justificacion": f"📚 Memoria ({recordado['veces']} casos similares)",
                "confianza": confianza
            })
        
        # --- 2. Buscar en HISTÓRICO (coincidencia fuzzy) ---
        matches = process.extract(desc_f, historico_desc_list, limit=5, scorer=fuzz.token_set_ratio)
        
        mejores_coincidencias = []
        for val, score in matches:
            if score > 65:  # Umbral de confianza
                # Buscar el código real de esa cuenta
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                nombre_historico = row['AcctName']
                codigo_historico = None
                for code, name in catalogo_dict.items():
                    if name == nombre_historico:
                        codigo_historico = code
                        break
                
                if codigo_historico:
                    mejores_coincidencias.append({
                        'descripcion': val,
                        'codigo': codigo_historico,
                        'nombre': nombre_historico,
                        'score': score
                    })
        
        if mejores_coincidencias:
            mejor = mejores_coincidencias[0]
            codigo_sugerido = mejor['codigo']
            nombre_sugerido = mejor['nombre']
            es_correcta = (codigo_sugerido == codigo_actual)
            
            # Aprender esta asociación
            aprender(desc_f, codigo_sugerido, nombre_sugerido)
            
            return jsonify({
                "es_correcta": es_correcta,
                "codigo_sugerido": codigo_sugerido,
                "nombre_sugerido": nombre_sugerido,
                "justificacion": f"📚 Similar a: '{mejor['descripcion'][:50]}' ({mejor['score']:.0f}%)",
                "confianza": mejor['score'] / 100
            })
        
        # --- 3. Si no hay coincidencia, mantener código actual ---
        nombre_real = obtener_nombre_real(codigo_actual)
        if nombre_real:
            return jsonify({
                "es_correcta": True,
                "codigo_sugerido": codigo_actual,
                "nombre_sugerido": nombre_real,
                "justificacion": "✓ Sin coincidencias claras, se mantiene cuenta actual",
                "confianza": 0.5
            })
        else:
            return jsonify({
                "es_correcta": False,
                "codigo_sugerido": codigo_actual,
                "nombre_sugerido": nombre_actual,
                "justificacion": "⚠️ Código actual no encontrado en catálogo",
                "confianza": 0
            })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e), "es_correcta": False}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint para corregir manualmente"""
    try:
        data = request.json
        desc_f = data.get('descripcion', '')
        codigo_correcto = data.get('codigo_correcto', '')
        
        if not desc_f or not codigo_correcto:
            return jsonify({"error": "Faltan campos"}), 400
        
        nombre_real = obtener_nombre_real(codigo_correcto)
        if not nombre_real:
            return jsonify({"error": f"Código {codigo_correcto} no existe"}), 400
        
        aprender(desc_f, codigo_correcto, nombre_real)
        
        return jsonify({
            "status": "✅ Aprendizaje reforzado",
            "patron": extraer_patron(desc_f),
            "codigo": codigo_correcto,
            "nombre": nombre_real
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memoria')
def ver_memoria():
    return jsonify({
        "total_patrones": len(MEMORIA),
        "patrones": {k: v for k, v in list(MEMORIA.items())[:50]}
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "active",
        "patrones_aprendidos": len(MEMORIA),
        "historico_filas": len(historico_desc_list),
        "catalogo_cuentas": len(catalogo_dict)
    })

@app.route('/')
def index():
    return "API de Auditoría Contable activa. Endpoints: /auditar, /feedback, /memoria, /health"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

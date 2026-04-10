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

# --- CARGA DE CATÁLOGO ---
df_catalogo = pd.read_csv(
    'results_1.csv', 
    sep=';',
    names=['AcctCode', 'AcctName'],
    encoding='latin-1',
    on_bad_lines='skip'
)
catalogo_codigos = df_catalogo['AcctCode'].astype(str).tolist()

def obtener_nombre_desde_catalogo(codigo):
    """Busca el nombre REAL de la cuenta en el catálogo"""
    if not codigo:
        return None
    
    codigo_str = str(codigo).strip()
    
    # Búsqueda exacta
    fila = df_catalogo[df_catalogo['AcctCode'].astype(str) == codigo_str]
    if not fila.empty:
        return fila.iloc[0]['AcctName']
    
    # Búsqueda fuzzy
    match = process.extractOne(codigo_str, catalogo_codigos, scorer=fuzz.token_set_ratio)
    if match and match[1] > 80:
        fila = df_catalogo[df_catalogo['AcctCode'].astype(str) == match[0]]
        if not fila.empty:
            return fila.iloc[0]['AcctName']
    
    return None

def es_codigo_valido(codigo):
    """Código de cuenta debe ser 4-6 dígitos numéricos"""
    if not codigo:
        return False
    return bool(re.match(r'^\d{4,6}$', str(codigo).strip()))

def corregir_codigo_sugerido(codigo_sugerido, codigo_actual):
    """Corrige códigos alucinados (letras, muy cortos, etc.)"""
    if es_codigo_valido(codigo_sugerido):
        return codigo_sugerido
    
    # Buscar en catálogo por similitud
    if catalogo_codigos:
        match = process.extractOne(str(codigo_sugerido), catalogo_codigos, scorer=fuzz.token_set_ratio)
        if match and match[1] > 60:
            return match[0]
    
    # Si no hay match, usar el código actual
    return codigo_actual

# --- MEMORIA PERSISTENTE ---
MEMORIA_ARCHIVO = 'memoria.json'
MEMORIA = {}

def cargar_memoria():
    global MEMORIA
    if os.path.exists(MEMORIA_ARCHIVO):
        with open(MEMORIA_ARCHIVO, 'r', encoding='latin-1') as f:
            MEMORIA = json.load(f)
    print(f"📚 Memoria cargada: {len(MEMORIA)} patrones")

def guardar_memoria():
    with open(MEMORIA_ARCHIVO, 'w', encoding='latin-1') as f:
        json.dump(MEMORIA, f, indent=2)

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
            'aciertos': 1 if correcta else 0,
            'ultimo': datetime.now().isoformat()
        }
    else:
        MEMORIA[patron]['veces'] += 1
        if correcta:
            MEMORIA[patron]['aciertos'] += 1
        MEMORIA[patron]['ultimo'] = datetime.now().isoformat()
    
    guardar_memoria()

# --- CARGA DEL HISTÓRICO ---
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

# Inicializar
cargar_memoria()

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
            es_correcta = (recordado['codigo'] == codigo_actual)
            resultado = {
                "es_correcta": es_correcta,
                "codigo_sugerido": recordado['codigo'] if not es_correcta else codigo_actual,
                "nombre_sugerido": recordado['nombre'] if not es_correcta else nombre_actual,
                "justificacion": f"📚 Basado en {recordado['veces']} casos previos",
                "confianza": recordado['aciertos'] / recordado['veces']
            }
            return jsonify(resultado)
        
        # --- 2. Buscar contexto histórico ---
        m_h = process.extract(desc_f, historico_desc_list, limit=2, scorer=fuzz.token_set_ratio)
        ctx = ""
        for val, score in m_h:
            if score > 40:
                row = df_historico[df_historico['Dscription'] == val].iloc[0]
                ctx += f"- {val[:50]} → {row['AcctName']}\n"
        
        # --- 3. Llamar a IA ---
        prompt = f"""Valida esta factura de compra:

Descripción: "{desc_f[:100]}"
Cuenta actual: {codigo_actual} - {nombre_actual}

Referencias históricas:
{ctx}

Responde SOLO en JSON:
{{"es_correcta": true/false, "codigo_sugerido": "código de 4-6 dígitos", "nombre_sugerido": "nombre de la cuenta", "justificacion": "explicación breve"}}"""

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300
        )
        
        resultado = json.loads(res.choices[0].message.content)
        
        # --- 4. Validar y corregir el código sugerido ---
        codigo_sugerido_ia = str(resultado.get('codigo_sugerido', codigo_actual))
        nombre_sugerido_ia = resultado.get('nombre_sugerido', nombre_actual)
        
        # Verificar si el código existe en el catálogo
        nombre_real = obtener_nombre_desde_catalogo(codigo_sugerido_ia)
        
        if nombre_real:
            # El código existe, usar el nombre real del catálogo
            resultado['codigo_sugerido'] = codigo_sugerido_ia
            resultado['nombre_sugerido'] = nombre_real
        else:
            # El código no existe, probablemente alucinación
            # Buscar si hay algún código similar en el catálogo
            codigo_corregido = corregir_codigo_sugerido(codigo_sugerido_ia, codigo_actual)
            nombre_corregido = obtener_nombre_desde_catalogo(codigo_corregido)
            
            resultado['codigo_sugerido'] = codigo_corregido
            resultado['nombre_sugerido'] = nombre_corregido if nombre_corregido else nombre_actual
            resultado['justificacion'] = f"⚠️ El código '{codigo_sugerido_ia}' no es válido. {resultado.get('justificacion', '')}"
        
        # --- 5. Aprender de esta consulta ---
        aprender(desc_f, resultado['codigo_sugerido'], resultado['nombre_sugerido'], resultado.get('es_correcta', False))
        
        return jsonify(resultado)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e), "es_correcta": False}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint para corregir manualmente cuando la IA falla"""
    try:
        data = request.json
        desc_f = data.get('descripcion', '')
        codigo_correcto = data.get('codigo_correcto', '')
        nombre_correcto = data.get('nombre_correcto', '')
        
        if not desc_f or not codigo_correcto:
            return jsonify({"error": "Faltan campos requeridos"}), 400
        
        # Forzar aprendizaje con corrección manual
        aprender(desc_f, codigo_correcto, nombre_correcto, True)
        
        return jsonify({
            "status": "✅ Aprendizaje reforzado",
            "patron": extraer_patron(desc_f),
            "codigo": codigo_correcto,
            "nombre": nombre_correcto
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memoria', methods=['GET'])
def ver_memoria():
    """Ver todos los patrones aprendidos"""
    return jsonify({
        "total_patrones": len(MEMORIA),
        "patrones": {k: v for k, v in list(MEMORIA.items())[:50]}  # Mostrar primeros 50
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "active",
        "memoria_patrones": len(MEMORIA),
        "catalogo_cuentas": len(catalogo_codigos),
        "historico_filas": len(historico_desc_list)
    })

@app.route('/')
def index():
    return "API de Auditoría Contable activa. Endpoints: /auditar, /feedback, /memoria, /health"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

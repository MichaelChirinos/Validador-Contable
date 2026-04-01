import os
from flask import Flask, request, jsonify
from thefuzz import process
from groq import Groq

app = Flask(__name__)

# Configuración de Groq - Render leerá esto de sus "Environment Variables"
# Localmente puedes usar: os.environ.get("GROQ_API_KEY", "tu_key_de_prueba")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def obtener_veredicto_ia(descripcion, cuenta_actual, opciones_reducidas):
    prompt = f'''
Eres un Auditor de Sistemas Contables experto en SAP Business One. 
Tu objetivo es detectar inconsistencias entre la descripción de una transacción y la cuenta contable asignada, utilizando el Plan Contable General Empresarial (PCGE).

**CONTEXTO:**
Los registros que estás auditando corresponden a **facturas de compras**. La descripción que recibes es el concepto de cada ítem o servicio facturado por el proveedor.

DATOS DE ENTRADA:
- Descripción del Ítem: "{descripcion}"
- Cuenta Asignada Actualmente: {cuenta_actual['AcctCode']} - {cuenta_actual['AcctName']}

CATÁLOGO DE CUENTAS DE DETALLE (Sugerencias similares):
{opciones_reducidas}

INSTRUCCIONES UNIVERSALES DE VALIDACIÓN:
1. CONSISTENCIA SEMÁNTICA: Compara la naturaleza del texto (¿Es un producto físico, un servicio, un impuesto, un movimiento bancario o una provisión?) con la naturaleza de la cuenta asignada.
2. NIVEL DE REGISTRO: Solo son válidas las cuentas de último nivel (detalle). Si la cuenta actual es genérica y existe una específica que nombre el concepto exacto de la descripción, sugiere el cambio.
3. DETECCIÓN DE "CUENTAS COMODÍN": Si la cuenta actual es una cuenta transitoria o de 'varios' (ej. FACTURAS POR RECIBIR, CUENTAS POR PAGAR, CUENTAS PUENTE) y en el catálogo aparece la cuenta de destino real que corresponde a la naturaleza del ítem (producto, servicio, activo, etc.), márcalo como incorrecto.
4. JERARQUÍA DE COINCIDENCIA: Prioriza siempre la cuenta que contenga la mayor cantidad de palabras clave de la descripción en su nombre.

FORMATO DE SALIDA (JSON ESTRICTO):
{{
    "es_correcta": true/false,
    "codigo_sugerido": "XXXXXX",
    "nombre_sugerido": "XXXXXX",
    "justificacion": "Breve análisis de la discrepancia lógica encontrada."
}}
'''
      
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message.content
    except Exception as e:
        return str({"error_ia": str(e)})

@app.route('/auditar', methods=['POST'])
def auditar():
    try:
        data = request.json
        desc_sql = data.get('descripcion_sql', '')
        cuenta_actual = data.get('cuenta_actual', {})
        catalogo_completo = data.get('catalogo', [])

        if not catalogo_completo or not desc_sql:
            return jsonify({"error": "Faltan datos"}), 400

        # --- PASO 1: FILTRAR SOLO CUENTAS DE DETALLE (4+ dígitos) ---
        # Corregido: cambiado de 6 a 4 dígitos como mínimo
        catalogo_detalle = [item for item in catalogo_completo if len(str(item.get('AcctCode', ''))) >= 4]
        
        # --- PASO 2: FUZZY MATCHING SOBRE EL CATÁLOGO FILTRADO ---
        nombres_cat = [item['AcctName'] for item in catalogo_detalle]
        matches = process.extract(desc_sql, nombres_cat, limit=10)
        
        top_10_nombres = [m[0] for m in matches]
        opciones_reducidas = [item for item in catalogo_detalle if item['AcctName'] in top_10_nombres]

        # --- PASO 3: CONSULTA A GROQ ---
        resultado_ia = obtener_veredicto_ia(desc_sql, cuenta_actual, opciones_reducidas)
        return resultado_ia

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta de prueba para verificar que el servicio está vivo en Render
@app.route('/')
def health_check():
    return "API de Auditoría Contable activa y lista."

if __name__ == '__main__':
    # Importante para Render: leer el puerto que le asignan
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

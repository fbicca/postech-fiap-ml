from flask import Flask, render_template, request, jsonify, send_file, session, send_from_directory, url_for
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from typing import Optional
import os, io, re, random, tempfile, subprocess
import requests
from validation import *
from anamnese import *
from datetime import datetime

# ------------------------- Inicialização -------------------------
load_dotenv()

# -------- Flask --------
app = Flask(__name__)
# ✅ caminho ABSOLUTO para a pasta de uploads
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "alura")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads
app.config["JSON_AS_ASCII"] = False  # JSON UTF-8 (sem \u00e9)

# Persistência simples em memória
db_memory = {}

# Configurar CORS
CORS(
    app,
    origins=["http://localhost:5001", "http://127.0.0.1:5001"],
    supports_credentials=True,
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)

# ------------------------- Integração com API de Predição -------------------------
API_PREDICT_HEART = os.getenv("API_PREDICT_HEART", "http://localhost:8001/predict")
API_PREDICT_PNEUMONIA = os.getenv("API_PREDICT_PNEUMONIA", "http://localhost:8002/predict")

def _build_api_payload(session):
    """Monta o payload esperado pela API a partir dos valores normalizados já salvos na sessão."""
    def to_int(x):
        try:
            return int(x)
        except:
            return None

    def to_float(x):
        try:
            return float(str(x).replace(",", "."))
        except:
            return None

    payload = {
        "Age": to_int(db_memory["idade"]),
        "Sex": db_memory["sexo"],
        "ChestPainType": db_memory["chestpain_type"],
        "RestingBP": to_float(db_memory["restingbp"]),
        "Cholesterol": to_int(db_memory["cholesterol"]),
        "FastingBS": 1 if db_memory["fastingbs"] in (1, "1", True, "sim") else 0,
        "RestingECG": db_memory["restingecg"],
        "MaxHR": to_int(db_memory["maxhr"]),
        "Exang": 1 if db_memory["exang"] in (1, "1", True, "sim") else 0,
        "Oldpeak": to_float(db_memory["oldpeak"]),
        "ST_Slope": db_memory["st_slope"],
    }
    return payload

def _call_predict_api(payload: dict):
    try:
        resp = requests.post(API_PREDICT_HEART, json=payload, timeout=10)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as e:
        return False, f"Falha ao chamar a API em {API_PREDICT_HEART}: {e}"

def _call_pneumonia_api(file_path: str):
    """
    Envia a imagem por multipart/form-data para a API de Pneumonia — sua FastAPI em /predict
    com resposta no formato: {top_class: str, top_prob: float, probs: {...}}
    """
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
            resp = requests.post(API_PREDICT_PNEUMONIA, files=files, timeout=20)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as e:
        return False, f"Falha ao chamar a API em {API_PREDICT_PNEUMONIA}: {e}"

def _falsy_no(s: str) -> bool:
    return (s or "").strip().lower() in {"não", "nao", "n", "cancelar", "trocar", "reenviar"}

def _save_upload(file_storage):
    """Salva o arquivo do campo 'file' e retorna (filename, url, path, bubble_html)."""
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        raise ValueError("Nome de arquivo inválido.")
    filename = unique_name(UPLOAD_FOLDER, filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(save_path)
    file_url = url_for("serve_upload", filename=filename, _external=True)

    # ✅ Mensagem "✅ Upload concluído" DENTRO da bolha do usuário
    if is_image_filename(filename):
        bubble_html = f"""
        <figure style="margin:0">
          <img src="{file_url}" alt="{escape_html(filename)}"
               style="max-width:240px;max-height:200px;border-radius:10px;border:1px solid #e0e0e0"/>
          <figcaption style="margin-top:6px;font-size:0.95rem;line-height:1.35">
            ✅ Upload concluído: <strong>{escape_html(filename)}</strong>
          </figcaption>
        </figure>
        """.strip()
    else:
        bubble_html = f"""<div>✅ Upload concluído:
            <a href="{file_url}" target="_blank" rel="noopener">{escape_html(filename)}</a></div>"""

    return filename, file_url, save_path, bubble_html

def unique_name(folder: str, filename: str) -> str:
    """
    Evita sobrescrita: se já existir, acrescenta timestamp.
    """
    base, ext = os.path.splitext(filename)
    candidate = filename
    path = os.path.join(folder, candidate)
    if not os.path.exists(path):
        return candidate
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}_{ts}{ext}"

def gerar_explicacao(payload: dict, label: str, openai_explanation: Optional[str] = None) -> str:
    """
    Gera uma explicação legível do diagnóstico.
    Prioriza a explicação da OpenAI se fornecida, caso contrário usa explicação local baseada em regras.
    
    Args:
        payload: Dados do paciente enviados à API
        label: Label da predição (ALTO_RISCO ou BAIXO_RISCO)
        openai_explanation: Explicação gerada pela OpenAI (opcional)
    
    Returns:
        String formatada com a explicação do diagnóstico
    """
    # Se houver explicação da OpenAI, usar ela
    # Verificar se não é None e se não é string vazia após strip
    if openai_explanation is not None and str(openai_explanation).strip():
        return "\n🤖 *Explicação do Diagnóstico*\n" + str(openai_explanation).strip() + "\n"
    
    # Fallback: explicação local baseada em regras
    try:
        idade  = payload.get("Age") or 0
        hr     = payload.get("MaxHR") or 0
        slope  = (payload.get("ST_Slope") or "").lower()
        chest  = (payload.get("ChestPainType") or "").lower()
        fbs    = payload.get("FastingBS")
        ecg    = (payload.get("RestingECG") or "").lower()
        oldpk  = payload.get("Oldpeak") or 0.0
        exang  = payload.get("Exang")

        motivos = []
        if label == "ALTO_RISCO":
            if isinstance(idade, (int, float)) and idade >= 55: motivos.append("idade avançada")
            if isinstance(hr, (int, float)) and hr < 100: motivos.append("HR baixo (<100)")
            if fbs == 1: motivos.append("jejum alterado")
            if "flat" in slope: motivos.append("ST plano")
            if "down" in slope: motivos.append("ST descendente")
            if "asy" in chest: motivos.append("assintomático")
            if "lvh" in ecg: motivos.append("ECG com hipertrofia")
            if exang in (1, "1", True, "sim"): motivos.append("esforço com angina")
            try:
                if float(oldpk) >= 2.0: motivos.append("oldpeak alto")
            except Exception:
                pass
            texto = ", ".join(motivos) or "características semelhantes às observadas em pacientes com doença cardíaca"
            return "\n➡️ Explicação:\n" + texto + " → o modelo tende a classificar como alto risco.\n\n"
        else:
            if isinstance(idade, (int, float)) and idade < 50: motivos.append("idade jovem")
            if isinstance(hr, (int, float)) and hr > 140: motivos.append("HR elevado (>140)")
            if fbs == 0: motivos.append("jejum normal")
            if "up" in slope: motivos.append("ST ascendente")
            if ("ata" in chest) or ("nap" in chest): motivos.append("dor anginosa atípica/não anginosa")
            if "normal" in ecg: motivos.append("ECG normal")
            if exang in (0, "0", False, "não", "nao"): motivos.append("sem angina ao esforço")
            try:
                if float(oldpk) <= 0.2: motivos.append("oldpeak baixo")
            except Exception:
                pass
            texto = ", ".join(motivos) or "padrão compatível com baixo risco de doença cardíaca"
            return "\n➡️ Explicação:\n" + texto + " → o modelo tende a classificar como baixo risco.\n\n"
    except Exception as e:
        return f"\n➡️ Explicação automática não gerada ({e}).\n\n"
    

#------------------------- Utilitários Upload -------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")

def is_image_filename(name: str) -> bool:
    try:
        return name.lower().endswith(IMAGE_EXTS)
    except:
        return False

def escape_html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )

def unique_name(folder: str, filename: str) -> str:
    """
    Evita sobrescrita: se já existir, acrescenta timestamp.
    """
    base, ext = os.path.splitext(filename)
    candidate = filename
    path = os.path.join(folder, candidate)
    if not os.path.exists(path):
        return candidate
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}_{ts}{ext}"

def _truthy_yes(s: str) -> bool:
    return (s or "").strip().lower() in {"sim", "s", "ok", "confirmo", "confirmar", "sim, enviar", "enviar"}


# ------------------------- Lógica do Chatbot -------------------------
def greet_and_menu():
    return jsonify({
        "msg": (
            "👋 Olá! Seja bem-vindo(a) à avaliação de saúde assistida.\n\n"
            "Este assistente foi desenvolvido para ajudá-lo(a) a estimar, de forma simples e segura, dois tipos de avaliações clínicas:\n\n"
            " 1️⃣ Avaliação de Risco Cardiovascular — baseada em informações clínicas que permitem identificar o risco de doenças cardíacas.\n"
            " 2️⃣ Avaliação de Quadro de Pneumonia — realizada a partir da análise de uma imagem de raio X de tórax.\n\n"
            "Por favor, escolha uma das opções abaixo para começar:\n"
            "👉 Responda com o número correspondente:\n\n"
            "1 — Avaliação de Risco Cardiovascular\n"
            "2 — Avaliação de Quadro de Pneumonia\n"
            "3 — Encerrar"
        ),
        "type_conversation": "await_service"
    })

# ------------------------- Normalização do retorno da API de Pneumonia -------------------------
def _pick_first(d: dict, keys: list):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None

def _coerce_class(v):
    """Normaliza classe para 'PNEUMONIA'|'NORMAL' a partir de string/int/bool."""
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return "PNEUMONIA" if float(v) >= 0.5 else "NORMAL"
    s = str(v).strip().lower()
    if "pneumonia" in s or s in {"positive", "pos", "1", "true"}:
        return "PNEUMONIA"
    if "normal" in s or s in {"negative", "neg", "0", "false"}:
        return "NORMAL"
    return None

def _dig(obj, *path_keys):
    cur = obj
    for k in path_keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur

def _normalize_pneumonia_result(res: dict):
    """
    Extrai (classe, prob, warnings) de formatos variados:
      - classe: prediction/label/class/diagnosis/result/pred/output/category...
      - prob: probability/score/confidence/prob_positive/proba...
      - pode estar em res, res['data'], res['result'], res['output'], ...
    """
    candidates = [res]
    for k in ("data", "result", "output", "payload", "response"):
        sub = _dig(res, k)
        if isinstance(sub, dict):
            candidates.append(sub)

    pred = prob = warns = None
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        raw_class = _pick_first(cand, [
            "prediction", "label", "class", "diagnosis", "result", "pred", "output", "category"
        ])
        if raw_class is not None and pred is None:
            pred = _coerce_class(raw_class)

        raw_prob = _pick_first(cand, [
            "probability", "score", "confidence", "prob", "proba",
            "pneumonia_probability", "prob_positive", "positive_probability"
        ])
        if raw_prob is not None and prob is None:
            try:
                prob = float(raw_prob)
                if prob > 1.0 and prob <= 100.0:
                    prob = prob / 100.0
            except Exception:
                pass

        raw_warn = _pick_first(cand, ["warnings", "warning", "messages", "message"])
        if raw_warn is not None and warns is None:
            if isinstance(raw_warn, (list, tuple)):
                warns = list(raw_warn)
            else:
                warns = [str(raw_warn)]

    if pred is None and isinstance(res, list) and res:
        pred = _coerce_class(res[0])

    return pred, prob, (warns or [])

# ------------------------- Formatação final (top_class/top_prob) -------------------------
def _format_pneumonia_message(result: dict):
    """
    Monta a mensagem no formato solicitado, preferindo 'top_class' e 'top_prob'.
    Se não existirem, usa a normalização (_normalize_pneumonia_result).
    - Quando NORMAL → 'ℹ️ Exame indicativo de NORMALIDADE, ausência de PNEUMONIA'
    - Caso contrário → 'ℹ️ Exame indicativo para <classe>'
    """
    top_class = result.get("top_class")
    top_prob  = result.get("top_prob")

    # Fallback (para formatos alternativos)
    if top_class is None or top_prob is None:
        pred, prob, _warnings = _normalize_pneumonia_result(result)
        if top_class is None:
            top_class = pred if pred else "—"
        if top_prob is None:
            top_prob = prob

    # Probabilidade como 90.61%
    if isinstance(top_prob, (int, float)):
        prob_str = f"{top_prob:.2%}"
    else:
        prob_str = str(top_prob) if top_prob is not None else "—"

    cls_norm = (top_class or "").strip().upper()
    if cls_norm == "NORMAL":
        linha_cls = f"ℹ️ Exame indicativo de NORMALIDADE\n👉 Probabilidade p/ Normalidade: {prob_str}"
    else:
        linha_cls = f"ℹ️ Exame indicativo para {top_class or '—'}\n👉 Probabilidade p/ Pneumonia: {prob_str}"

    linhas = [
        "🔬 *Resultado da Análise de Pneumonia*",
        "",
        linha_cls,
        "",
        "Digite 'menu' para voltar ao início, ou '2' para analisar outra radiografia."
    ]
    return "\n".join(linhas)

 #------------------------- Upload (legado/compat) -------------------------
@app.post("/upload")
def upload_file():
    """
    Mantido para compatibilidade com versões antigas.
    Para o novo comportamento (enviar só quando clicar em 'Enviar'), use **POST /chat_send**.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado (campo 'file' ausente)."}), 400

        f = request.files["file"]
        if not f or f.filename.strip() == "":
            return jsonify({"error": "Nome de arquivo inválido."}), 400

        filename, file_url, save_path, bubble_html = _save_upload(f)

        # ✅ Guardar a última imagem no "estado" para o fluxo de confirmação
        db_memory["last_xray_filename"] = filename
        db_memory["last_xray_url"] = file_url
        db_memory["last_xray_path"] = save_path

        return jsonify({
            "message": "Upload realizado com sucesso!",
            "filename": filename,
            "url": file_url,
            "bubble_html": bubble_html
        }), 200

    except Exception as e:
        return jsonify({"error": f"Falha ao salvar arquivo: {e}"}), 500

# ✅ rota para servir arquivos enviados
@app.get("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)

###########################################################################################
# árvore do chatbot
###########################################################################################
@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("msg") or "").strip()
    type_conversation = data.get("type_conversation")
    uploaded_filename = (data.get("uploaded_filename") or "").strip()
    uploaded_url = (data.get("uploaded_url") or "").strip()

    print(f"Chatbot - user_msg: {user_msg}")
    print(f"Chatbot - type_conversation: {type_conversation}")

    # Evita HTML do front cair como mensagem
    if user_msg.startswith("<") or "</" in user_msg:
        user_msg = ""

    low = user_msg.lower()

    # Estado inicial robusto
    if not type_conversation:
        type_conversation = "await_service"
        print("Primeiro turno sem estado")

    # Reset para menu
    if low in {"menu", "inicio", "início", "recomeçar"}:
        print("reset state #2")
        return greet_and_menu()

    # Menu inicial
    if type_conversation == "await_service":
        if low in {"1", "um"}:
            return jsonify({
                "msg": "Perfeito! 👍\nAgora, por favor, informe a idade do paciente (em anos completos).",
                "type_conversation": "await_age"
            })
        elif low in {2, "2", "dois"}:
            return jsonify({
                "msg": (
                    "Perfeito! 🙌\nAgora, por favor, selecione a imagem do exame que deseja analisar."
                ),
                "type_conversation": "await_pneumonia_image",
                "ui": {"enable_upload": True, "auto_open": True}
            })
        elif low in {"encerrar", "fim"}:
            return jsonify({
                "msg": "✅ Entendido!\nO atendimento foi encerrado.\n\n💬 Agradecemos seu tempo e confiança. Cuide bem do seu coração! ❤️\n\nAté logo!",
                "type_conversation": "await_service"
            })
        else:
            return jsonify({
                "msg": (
                    "Por favor, escolha uma das opções abaixo para começar:\n\n"
                    "👉 Responda com o número correspondente:\n\n"
                    "1 — Avaliação de Risco Cardiovascular\n"
                    "2 — Avaliação de Quadro de Pneumonia\n"
                    "3 — Encerrar"
                ),
                "type_conversation": "await_service"
            })

    # ===== FLUXO PNEUMONIA (compat): se front mandar uploaded_filename aqui =====
    if type_conversation == "await_pneumonia_image":
        if uploaded_filename:
            db_memory["last_xray_filename"] = uploaded_filename
            if uploaded_url:
                db_memory["last_xray_url"] = uploaded_url
            possible_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)
            if os.path.exists(possible_path):
                db_memory["last_xray_path"] = possible_path

            fname = db_memory.get("last_xray_filename")
            return {
                "msg": (
                    f"✅ Upload concluído: {fname}\n\n"
                    "Deseja enviar esta imagem para análise de Pneumonia? 😉\n\n"
                    "👉 Responda 'SIM' para confirmar ou 'NÃO' para reenviar outra imagem."
                ),
                "type_conversation": "await_pneumonia_confirm",
                "ui": {"enable_upload": True, "auto_open": False}
            }

        return {
            "msg": "👉 Aguardando a imagem. Selecione o arquivo que deseja analisar.",
            "type_conversation": "await_pneumonia_image",
            "ui": {"enable_upload": True, "auto_open": True}
        }

    if type_conversation == "await_pneumonia_confirm":
        if _truthy_yes(low):
            img_path = db_memory.get("last_xray_path")
            if not img_path or not os.path.exists(img_path):
                return {
                    "msg": (
                        "❌ Não localizei a imagem para análise.\n"
                        "Envie novamente a imagem da radiografia que deseja avaliar."
                    ),
                    "type_conversation": "await_pneumonia_image",
                    "ui": {"enable_upload": True, "auto_open": True}
                }

            ok, result = _call_pneumonia_api(img_path)
            if ok:
                # ✅ Mesmo formato padronizado aqui
                return {
                    "msg": _format_pneumonia_message(result),
                    "type_conversation": "await_service"
                }
            else:
                return {
                    "msg": f"❌ {result}\n\nTente novamente enviando a imagem outra vez.",
                    "type_conversation": "await_pneumonia_image",
                    "ui": {"enable_upload": True, "auto_open": True}
                }

        if _falsy_no(low):
            return {
                "msg": (
                    "Sem problemas! 😊\n"
                    "Envie uma nova radiografia de tórax quando estiver pronto."
                ),
                "type_conversation": "await_pneumonia_image",
                "ui": {"enable_upload": True, "auto_open": True}
            }

        return {
            "msg": "👉 Por favor, responda 'SIM' para analisar esta imagem ou 'NÃO' para reenviar outra.",
            "type_conversation": "await_pneumonia_confirm"
        }



    # ===== FLUXO CARDIO =====
    # Idade
    if type_conversation == "await_age":
        print("await_age - processando idade")
        resultado = valida_idade(low)
        if resultado is True:
            db_memory["idade"] = int(low)
            return jsonify({
                "msg": f"Perfeito! 👏\nA idade registrada é {db_memory['idade']} anos.\n\nAgora, por favor, informe o sexo do paciente (Masculino ou Feminino).",
                "type_conversation": "await_sex"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_age"
            })

    # Sexo
    if type_conversation == "await_sex":
        ok, resultado = valida_sexo(low)
        if ok:
            db_memory["sexo"] = resultado
            return jsonify({
                "msg": (
                    f"Entendido! 👍\nSexo registrado: {'Masculino' if resultado == 'M' else 'Feminino'}.\n\n"
                    "Agora, por favor, informe se o paciente sente dor no peito. Se sim, escolha a opção que melhor descreve o tipo de dor:\n\n"
                    "💔 TA: Angina típica (dor típica de esforço)\n"
                    "💓 ATA: Angina atípica (dor atípica)\n"
                    "❤️ NAP: Dor não anginosa (não relacionada ao coração)\n"
                    "🚫 ASY: Assintomática (sem dor no peito)"
                ),
                "type_conversation": "await_chestpain"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_sex"
            })

    # ChestPainType
    if type_conversation == "await_chestpain":
        ok, resultado = valida_dor_no_peito(low)
        if ok:
            db_memory["chestpain_type"] = resultado
            return jsonify({
                "msg": (
                    f"Entendido! 👍\n Dor no peito registrada: {resultado}.\n\n"
                    "Agora, por favor, informe a pressão arterial em repouso (em mmHg)."
                ),
                "type_conversation": "await_restingbp"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_chestpain"
            })

    # RestingBP
    if type_conversation == "await_restingbp":
        resultado = valida_pressao(low)
        if resultado is True:
            db_memory["restingbp"] = int(low)
            return jsonify({
                "msg": f"Perfeito! 🙌\nPressão registrada: {db_memory['restingbp']} mmHg.\n\nAgora, por favor, informe o **nível de colesterol total** (em **mg/dL**).",
                "type_conversation": "await_cholesterol"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_restingbp"
            })

    # Colesterol
    if type_conversation == "await_cholesterol":
        resultado = valida_colesterol(low)
        if resultado is True:
            db_memory["cholesterol"] = int(low)
            return jsonify({
                "msg": (
                    f"Ótimo! 🙌 \nColesterol registrado: {db_memory['cholesterol']} mg/dL.\n\n"
                    "Agora, por favor, informe se o paciente estava em jejum (FastingBS).\n👉 Responda 'sim' ou 'não'."
                ),
                "type_conversation": "await_fastingbs"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_cholesterol"
            })

    # FastingBS
    if type_conversation == "await_fastingbs":
        ok, resultado = valida_jejum(low)
        if ok:
            db_memory["fastingbs"] = resultado
            return jsonify({
                "msg": (
                    f"Entendido! 👍\nPaciente {'estava' if resultado == 1 else 'não estava'} em jejum.\n\n"
                    "Agora, por favor, informe o resultado do eletrocardiograma em repouso (RestingECG).\n\n"
                    "As opções são:\n"
                    "🩺 Normal\n"
                    "⚡ ST-T wave abnormality\n"
                    "❤️ LVH Left ventricular hypertrophy"
                ),
                "type_conversation": "await_ecg"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_fastingbs"
            })

    # RestingECG
    if type_conversation == "await_ecg":
        ok, resultado = valida_ecg(low)
        if ok:
            db_memory["restingecg"] = resultado
            return jsonify({
                "msg": (
                    f"Perfeito! 💓\nResultado do ECG: {db_memory['restingecg']}.\n\n"
                    "Agora, por favor, informe a frequência cardíaca máxima atingida (MaxHR), em batimentos por minuto (bpm)."
                ),
                "type_conversation": "await_maxhr"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_ecg"
            })

    # MaxHR
    if type_conversation == "await_maxhr":
        ok, resultado = valida_maxhr(low)
        if ok:
            db_memory["maxhr"] = resultado
            return jsonify({
                "msg": (
                    f"Excelente! 🩺\nFrequência cardíaca máxima: {db_memory['maxhr']} bpm.\n\n"
                    "Agora, por favor, informe se o paciente apresentou angina induzida por exercício (Exang).\n👉 Responda 'sim' ou 'não'."
                ),
                "type_conversation": "await_exang"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_maxhr"
            })

    # Exang
    if type_conversation == "await_exang":
        ok, resultado = valida_exang(low)
        if ok:
            db_memory["exang"] = resultado
            return jsonify({
                "msg": (
                    f"Entendido 👍\nO paciente {'APRESENTOU' if resultado == 1 else 'NÃO APRESENTOU'} Angina Induzida durante o exercício.\n\n"
                    "Agora, por favor, informe o valor da depressão do segmento ST (Oldpeak), em relação ao repouso.\n"
                    "👉 Informe um número entre 0.0 e 10.0\n"
                ),
                "type_conversation": "await_oldpeak"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_exang"
            })

    # Oldpeak
    if type_conversation == "await_oldpeak":
        ok, resultado = valida_oldpeak(low)
        if ok:
            db_memory["oldpeak"] = resultado
            return jsonify({
                "msg": (
                    f"Perfeito 👍\nValor de Oldpeak registrado: {db_memory['oldpeak']} mV.\n\n"
                    "Agora, por favor, informe a inclinação do segmento ST (Slope):\n"
                    "📈 Up → crescente\n"
                    "➖ Flat → plano\n"
                    "📉 Down → decrescente"
                ),
                "type_conversation": "await_slope"
            })
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_oldpeak"
            })

    # Slope (finaliza a coleta e mostra resumo)
    if type_conversation == "await_slope":
        ok, resultado = valida_slope(low)
        if ok:
            db_memory["st_slope"] = resultado
            resumo = montar_resumo(db_memory)  # assume que retorna dict com msg e type_conversation
            # Se montar_resumo já entrega dict no formato esperado:
            if isinstance(resumo, dict):
                return jsonify(resumo)
            # Caso contrário, empacota:
            return jsonify({"msg": str(resumo), "type_conversation": "confirm_summary"})
        else:
            return jsonify({
                "msg": f"{resultado}",
                "type_conversation": "await_slope"
            })

    # Confirmação final
    if type_conversation == "confirm_summary":
        if low in {"sim", "confirmo", "ok"}:
            payload = _build_api_payload(db_memory)
            ok_api, result = _call_predict_api(payload)
            if ok_api:
                pred = result.get("prediction")
                label = result.get("label")
                prob = result.get("probability_positive")
                warnings = result.get("warnings") or []
                # Capturar explicações da OpenAI se disponíveis
                openai_explanation = result.get("explanation")  # Compatibilidade
                openai_explanation_patient = result.get("explanation_patient")
                openai_explanation_professional = result.get("explanation_professional")
                
                # Debug: verificar se as explicações estão sendo recebidas
                print(f"[DEBUG] Explicação recebida da API: {repr(openai_explanation)}")
                print(f"[DEBUG] Explicação paciente: {repr(openai_explanation_patient)}")
                print(f"[DEBUG] Explicação profissional: {repr(openai_explanation_professional)}")
                print(f"[DEBUG] Chaves no result: {list(result.keys())}")
                
                linhas = ["🔮 *Resultado da Predição*"]

                if str(label).strip().upper() in ["ALTO_RISCO", "ALTO RISCO", "1"]:
                    linhas.append("- Classe: 🔴 ALTO RISCO CARDÍACO")
                else:
                    linhas.append("- Classe: 🟢 BAIXO RISCO CARDÍACO")

                linhas.append(
                    f"- Probabilidade de classe positiva: {prob:.2%}"
                    if isinstance(prob, (int, float))
                    else f"- Probabilidade: {prob}"
                )
                if warnings:
                    linhas.append("\n⚠️ Avisos:\n " + "; ".join(warnings))
                
                # Usar explicação da OpenAI se disponível, senão usar explicação local
                # Priorizar explanation_patient se disponível, senão usar explanation (compatibilidade)
                explanation_to_use = openai_explanation_patient or openai_explanation
                
                if explanation_to_use and str(explanation_to_use).strip():
                    # Exibir explicação para o paciente
                    linhas.append("\n🤖 *Explicação para o Paciente*")
                    linhas.append(str(explanation_to_use).strip())
                    
                    # Se houver explicação profissional, adicionar também
                    if openai_explanation_professional and str(openai_explanation_professional).strip():
                        linhas.append("\n\n👨‍⚕️ *Análise Técnica para o Profissional*")
                        linhas.append(str(openai_explanation_professional).strip())
                else:
                    # Fallback para explicação local
                    explicacao_texto = gerar_explicacao(payload, label, None)
                    linhas.append(explicacao_texto)
                    
                    # Nota informativa se não há explicação da OpenAI
                    linhas.append("\n💡 *Nota*: Para receber explicações detalhadas geradas por IA, configure a OPENAI_API_KEY no servidor da API.")
                
                linhas.append("\nDigite 'sim' para iniciar novo atendimento ou 'não' para encerrar.")
                return jsonify({
                    "msg": "\n".join(linhas),
                    "type_conversation": "await_service"
                })
            else:
                return jsonify({
                    "msg": f"❌ {result}\n\nDigite 'sim' para tentar novamente ou 'não' para encerrar.",
                    "type_conversation": "await_service"
                })
        elif low in {"não", "nao"}:
            return greet_and_menu()
        else:
            return jsonify({
                "msg": "Por favor, responda 'sim' para confirmar e enviar à API, ou 'não' para recomeçar.",
                "type_conversation": "confirm_summary"
            })

    # Fallback absoluto (nunca sair sem retorno)
    return jsonify({
        "msg": (
            "Não entendi sua mensagem. Vamos recomeçar?\n\n"
            "👉 Responda com o número correspondente:\n"
            "1 — Avaliação de Risco Cardiovascular\n"
            "2 — Avaliação de Quadro de Pneumonia\n"
            "3 — Encerrar"
        ),
        "type_conversation": "await_service"
    })

# ------------------------- Home -------------------------
@app.get("/")
def home():
    try:
        db_memory["state"] = "await_service"
        return render_template("index.html")
    except Exception:
        return "BotHealth backend ativo."

# ------------------------- Main -------------------------
if __name__ == "__main__":
    print("[BotHealth] iniciado.")
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "6000")))

from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

# =============================
# CONFIG
# =============================
IMG_SIZE = 224
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # Updated: allow up to 50MB

# =============================
# EMAIL CONFIG
# =============================
app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = 'gynogicalcancertesticularcance@gmail.com'
app.config['MAIL_PASSWORD']       = 'ovae tlil rqcg byzx'
app.config['MAIL_DEFAULT_SENDER'] = 'gynogicalcancertesticularcance@gmail.com'

mail = Mail(app)

# =============================
# LOAD MODEL
# =============================
MODEL_PATH  = "cancer_model_best.h5"
LABELS_PATH = "class_labels.npy"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"❌ Labels file not found: {LABELS_PATH}")

model         = load_model(MODEL_PATH)
class_indices = np.load(LABELS_PATH, allow_pickle=True).item()
labels        = {v: k for k, v in class_indices.items()}

print(f"✅ Model loaded: {MODEL_PATH}")
print(f"✅ Classes loaded: {len(class_indices)} classes")
print(f"   Classes: {list(class_indices.keys())}")


# =============================
# ALLOWED FILE EXTENSIONS
# FIXED: Allow all common image formats including WebP, BMP, TIFF, GIF, DICOM
# =============================
ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png",     # standard
    "webp",                   # WebP
    "bmp",                    # Bitmap
    "tiff", "tif",            # TIFF (common in medical imaging)
    "gif",                    # GIF
    "dcm",                    # DICOM (medical standard)
    "jfif",                   # JFIF variant of JPEG
    "pjpeg", "pjp",           # progressive JPEG
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================
# PREDICTION FUNCTION
# =============================
def predict_image(image_path):
    img       = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction      = model.predict(img_array)[0]
    predicted_class = int(np.argmax(prediction))
    confidence      = float(np.max(prediction))

    label = labels.get(predicted_class, "unknown/unknown")

    parts = label.split("/")
    if len(parts) == 2:
        cancer_type  = parts[0].strip().title()
        cancer_stage = parts[1].strip().title()
    else:
        cancer_type  = label.title()
        cancer_stage = "Unknown"

    if cancer_type.lower() == "normal":
        cancer_stage = "None"

    return {
        "type"             : cancer_type,
        "stage"            : cancer_stage,
        "confidence"       : round(confidence * 100, 2),
        "raw_confidence"   : confidence,
        "all_probabilities": prediction.tolist(),
        "raw_label"        : label
    }


# =============================
# ROUTES
# =============================
@app.route("/")
def index():
    return render_template("index.html")


# =============================
# PREDICT
# =============================
@app.route("/predict", methods=["POST"])
def predict():

    file   = request.files.get("image")
    gender = request.form.get("gender", "").strip().lower()

    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    # FIXED: now accepts all image types (jpg, png, webp, bmp, tiff, gif, dcm etc.)
    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type. Allowed formats: JPG, PNG, WebP, BMP, TIFF, GIF, DICOM."
        }), 400

    if gender not in ["male", "female"]:
        return jsonify({"error": "Please select gender"}), 400

    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = predict_image(filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": "Model prediction failed. Please try again."}), 500

    raw_label     = result["raw_label"].lower()
    confidence    = result["raw_confidence"]
    probabilities = result["all_probabilities"]

    # -------------------------
    # Confidence Check
    # -------------------------
    if confidence < 0.40:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": "Low confidence. Please upload a clearer scan."}), 400

    # -------------------------
    # Dominance Check
    # -------------------------
    sorted_probs = sorted(probabilities, reverse=True)
    if len(sorted_probs) > 1:
        gap = sorted_probs[0] - sorted_probs[1]
        if gap < 0.10:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Uncertain result. Please upload a clearer scan."}), 400

    # -------------------------
    # Gender Validation
    # -------------------------
    if "normal" not in raw_label:
        cancer_type_part = raw_label.split("/")[0]

        if gender == "male" and "testicular" not in cancer_type_part:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Scan does not match male cancer type (Testicular)."}), 400

        if gender == "female" and "gyno" not in cancer_type_part:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Scan does not match female cancer type (Gynaecological)."}), 400

    return jsonify({
        "type"      : result["type"],
        "stage"     : result["stage"],
        "confidence": result["confidence"]
    })


# =============================
# SEND EMAIL
# =============================
@app.route("/send-report", methods=["POST"])
def send_report():
    data = request.get_json()

    if not data:
        return jsonify({"success": False, "message": "No data received"}), 400

    email       = data.get("email", "").strip()
    report_text = data.get("report", "").strip()

    if not email or "@" not in email:
        return jsonify({"success": False, "message": "Invalid email address"}), 400

    if not report_text:
        return jsonify({"success": False, "message": "Report content is empty"}), 400

    try:
        msg = Message(
            subject   = "OncoCare AI Diagnostic Report",
            recipients= [email],
            body      = report_text
        )
        mail.send(msg)
        return jsonify({"success": True, "message": "Report sent successfully"})

    except Exception as e:
        print(f"[ERROR] Email failed: {e}")
        return jsonify({"success": False, "message": "Failed to send email. Check email config."}), 500


# =============================
# EXPORT PDF
# =============================
@app.route("/export-pdf", methods=["POST"])
def export_pdf():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    patient = data.get("patient", {})
    result  = data.get("result",  {})

    if not patient or not result:
        return jsonify({"error": "Missing patient or result data"}), 400

    patient_id = patient.get("id", str(uuid.uuid4()))
    filename   = f"Report_{patient_id}.pdf"
    filepath   = os.path.join("static", filename)

    try:
        doc      = SimpleDocTemplate(filepath, pagesize=A4)
        styles   = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("<b>ONCOCARE AI MEDICAL REPORT</b>", styles['Title']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
        elements.append(Paragraph(f"Patient ID : {patient.get('id',      'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Name       : {patient.get('name',    'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Age        : {patient.get('age',     'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Gender     : {patient.get('gender',  'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Contact    : {patient.get('contact', 'N/A')}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("<b>Diagnosis Result</b>", styles['Heading2']))
        elements.append(Paragraph(f"Cancer Type  : {result.get('type',       'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Cancer Stage : {result.get('stage',      'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Confidence   : {result.get('confidence', 'N/A')}%", styles['Normal']))
        elements.append(Spacer(1, 12))

        stage = result.get("stage", "").strip().lower()
        if stage == "critical":
            elements.append(Paragraph(
                "<b>⚠ CRITICAL STAGE DETECTED — Immediate Medical Attention Required</b>",
                styles['Normal']
            ))

        if result.get("type", "").lower() == "normal":
            elements.append(Paragraph(
                "<b>✅ No cancer detected. Patient appears healthy.</b>",
                styles['Normal']
            ))

        elements.append(Spacer(1, 24))
        elements.append(Paragraph(
            "<i>This report is generated by OncoCare AI and is not a substitute "
            "for professional medical advice.</i>",
            styles['Normal']
        ))

        doc.build(elements)

    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        return jsonify({"error": "PDF generation failed"}), 500

    return send_file(filepath, as_attachment=True)


# =============================
# GLOBAL ERROR HANDLERS
# =============================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)
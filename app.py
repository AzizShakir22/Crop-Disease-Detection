import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="LeafScan AI — Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Disease Info Database ─────────────────────────────
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "severity": "Moderate",
        "color": "#e8a838",
        "symptoms": "Small, water-soaked spots on leaves that turn brown with yellow halos.",
        "treatment": "Apply copper-based bactericides. Remove infected plant debris. Avoid overhead irrigation.",
        "prevention": "Use disease-free seeds. Practice crop rotation every 2–3 years.",
        "icon": "⚠️"
    },
    "Pepper__bell___healthy": {
        "severity": "Healthy",
        "color": "#2d9e6b",
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Maintain proper watering, fertilization and sunlight.",
        "icon": "✅"
    },
    "Potato___Early_blight": {
        "severity": "Moderate",
        "color": "#e8a838",
        "symptoms": "Dark brown spots with concentric rings (target-like) on older leaves.",
        "treatment": "Apply fungicides like chlorothalonil or mancozeb. Remove affected leaves immediately.",
        "prevention": "Ensure proper plant spacing for air circulation. Avoid wetting foliage.",
        "icon": "⚠️"
    },
    "Potato___Late_blight": {
        "severity": "Severe",
        "color": "#e84444",
        "symptoms": "Water-soaked lesions on leaves turning dark brown/black. White mold on undersides.",
        "treatment": "Apply systemic fungicides (metalaxyl). Destroy infected plants. Do not compost.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation. Monitor humidity levels.",
        "icon": "🚨"
    },
    "Potato___healthy": {
        "severity": "Healthy",
        "color": "#2d9e6b",
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Maintain proper soil nutrition and drainage.",
        "icon": "✅"
    },
    "Tomato_Bacterial_spot": {
        "severity": "Moderate",
        "color": "#e8a838",
        "symptoms": "Small, dark, water-soaked spots on leaves, stems, and fruits.",
        "treatment": "Copper-based sprays. Remove infected material. Avoid working with wet plants.",
        "prevention": "Use certified disease-free seeds. Rotate crops annually.",
        "icon": "⚠️"
    },
    "Tomato_Early_blight": {
        "severity": "Moderate",
        "color": "#e8a838",
        "symptoms": "Brown spots with concentric rings on lower/older leaves. Yellow halo around spots.",
        "treatment": "Apply fungicides (chlorothalonil, copper). Remove lower infected leaves.",
        "prevention": "Mulch around plants. Water at base. Ensure good air circulation.",
        "icon": "⚠️"
    },
    "Tomato_Late_blight": {
        "severity": "Severe",
        "color": "#e84444",
        "symptoms": "Greasy, dark lesions on leaves. Brown decay on stems and fruits.",
        "treatment": "Apply systemic fungicides immediately. Remove and destroy all infected material.",
        "prevention": "Avoid overhead watering. Use resistant varieties. Monitor forecasts.",
        "icon": "🚨"
    },
    "Tomato_Leaf_Mold": {
        "severity": "Moderate",
        "color": "#e8a838",
        "symptoms": "Yellow spots on upper leaf surface, olive-green/gray mold on underside.",
        "treatment": "Apply fungicides. Improve ventilation. Reduce humidity in greenhouses.",
        "prevention": "Plant resistant varieties. Ensure good air circulation. Avoid leaf wetness.",
        "icon": "⚠️"
    },
    "Tomato_Septoria_leaf_spot": {
        "severity": "Moderate",
        "color": "#e8a838",
        "symptoms": "Small circular spots with dark borders and light centers on lower leaves.",
        "treatment": "Apply fungicides. Remove heavily infected leaves. Avoid overhead watering.",
        "prevention": "Crop rotation. Stake plants for better airflow. Mulch soil.",
        "icon": "⚠️"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "severity": "Moderate",
        "color": "#e8a838",
        "symptoms": "Fine webbing on leaves. Tiny yellow/white spots (stippling). Leaves turn bronze.",
        "treatment": "Apply miticides or neem oil. Increase humidity. Introduce predatory mites.",
        "prevention": "Monitor regularly. Avoid dusty conditions. Keep plants well-watered.",
        "icon": "⚠️"
    },
    "Tomato__Target_Spot": {
        "severity": "Moderate",
        "color": "#e8a838",
        "symptoms": "Brown lesions with target-like concentric rings. Affects leaves, stems, fruits.",
        "treatment": "Apply fungicides. Remove infected debris. Improve drainage.",
        "prevention": "Avoid overcrowding. Use drip irrigation. Practice crop rotation.",
        "icon": "⚠️"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "severity": "Severe",
        "color": "#e84444",
        "symptoms": "Yellowing and upward curling of leaves. Stunted growth. Reduced fruit set.",
        "treatment": "No cure. Remove infected plants. Control whitefly vectors with insecticides.",
        "prevention": "Use resistant varieties. Control whiteflies. Use reflective mulches.",
        "icon": "🚨"
    },
    "Tomato__Tomato_mosaic_virus": {
        "severity": "Severe",
        "color": "#e84444",
        "symptoms": "Mosaic pattern of light/dark green on leaves. Leaf distortion. Stunted growth.",
        "treatment": "No cure. Remove and destroy infected plants. Disinfect tools.",
        "prevention": "Use virus-free seeds. Wash hands before handling. Control aphids.",
        "icon": "🚨"
    },
    "Tomato_healthy": {
        "severity": "Healthy",
        "color": "#2d9e6b",
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "treatment": "No treatment needed. Continue regular care.",
        "prevention": "Maintain regular watering, fertilization schedule and pest monitoring.",
        "icon": "✅"
    },
}

# ─── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0f0d;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Hero Section */
.hero {
    background: linear-gradient(135deg, #0a1a12 0%, #0f2318 50%, #0a1a12 100%);
    padding: 60px 60px 40px;
    border-bottom: 1px solid #1a3326;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(45,158,107,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(45,158,107,0.15);
    border: 1px solid rgba(45,158,107,0.3);
    color: #4ecca3;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 56px;
    font-weight: 700;
    color: #e8f5ee;
    line-height: 1.1;
    margin: 0 0 16px 0;
}

.hero h1 span {
    color: #4ecca3;
}

.hero p {
    font-size: 18px;
    color: #7aab90;
    max-width: 560px;
    line-height: 1.6;
    font-weight: 300;
}

.hero-stats {
    display: flex;
    gap: 40px;
    margin-top: 40px;
    padding-top: 32px;
    border-top: 1px solid #1a3326;
}

.hero-stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: #4ecca3;
}

.hero-stat-label {
    font-size: 13px;
    color: #4a7a62;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Main content */
.main-content {
    padding: 40px 60px;
    background: #0a0f0d;
}

/* Upload area */
.upload-label {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: #c8e6d4;
    margin-bottom: 8px;
    font-weight: 600;
}

.upload-hint {
    font-size: 14px;
    color: #4a7a62;
    margin-bottom: 20px;
}

/* Result card */
.result-card {
    background: #0f1f17;
    border: 1px solid #1a3326;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
}

.result-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 20px;
}

.result-icon {
    font-size: 36px;
    line-height: 1;
}

.result-disease {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 600;
    color: #e8f5ee;
    line-height: 1.2;
}

.result-crop {
    font-size: 13px;
    color: #4a7a62;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.05em;
    margin-bottom: 20px;
}

.severity-healthy { background: rgba(45,158,107,0.2); color: #4ecca3; border: 1px solid rgba(45,158,107,0.3); }
.severity-moderate { background: rgba(232,168,56,0.2); color: #f0c060; border: 1px solid rgba(232,168,56,0.3); }
.severity-severe { background: rgba(232,68,68,0.2); color: #f07070; border: 1px solid rgba(232,68,68,0.3); }

.conf-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.conf-label { font-size: 13px; color: #7aab90; width: 140px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conf-bar-bg { flex: 1; height: 4px; background: #1a3326; border-radius: 99px; overflow: hidden; }
.conf-bar { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
.conf-pct { font-size: 13px; color: #c8e6d4; font-weight: 500; width: 36px; text-align: right; }

/* Info sections */
.info-section {
    background: #0f1f17;
    border: 1px solid #1a3326;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 16px;
}

.info-section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a7a62;
    margin-bottom: 10px;
    font-weight: 500;
}

.info-section-body {
    font-size: 15px;
    color: #a8d4bc;
    line-height: 1.65;
    font-weight: 300;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #1a3326;
    margin: 32px 0;
}

/* Supported crops section */
.crops-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 16px;
}

.crop-chip {
    background: #0f1f17;
    border: 1px solid #1a3326;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #7aab90;
}

.crop-chip-name {
    font-weight: 500;
    color: #c8e6d4;
    margin-bottom: 2px;
}

/* Upload section styling override */
[data-testid="stFileUploader"] {
    background: #0f1f17 !important;
    border: 1.5px dashed #2d5c42 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #4ecca3 !important;
}

/* Image display */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid #1a3326 !important;
}

/* Metric overrides */
[data-testid="stMetric"] {
    background: #0f1f17 !important;
    border: 1px solid #1a3326 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: #4ecca3 !important;
    font-size: 28px !important;
}

[data-testid="stMetricLabel"] {
    color: #4a7a62 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #4ecca3 !important; }

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 600;
    color: #c8e6d4;
    margin-bottom: 6px;
}

.section-subtitle {
    font-size: 14px;
    color: #4a7a62;
    margin-bottom: 24px;
}

/* Footer */
.footer {
    background: #060d09;
    border-top: 1px solid #1a3326;
    padding: 28px 60px;
    text-align: center;
}

.footer p {
    font-size: 13px;
    color: #2d5c42;
}

.footer span { color: #4ecca3; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load('crop_disease_model.pth', map_location='cpu')
        class_names = checkpoint['class_names']
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.last_channel, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(class_names))
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, class_names, True
    except Exception as e:
        return None, None, False


def predict(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
    top5 = torch.topk(probs, min(5, len(class_names)))
    return [(class_names[i], probs[i].item()) for i in top5.indices]


def format_name(raw):
    parts = raw.replace('__', '_').replace('___', '_').split('_')
    cleaned = []
    for p in parts:
        if p and p not in ['Two', 'spotted']:
            cleaned.append(p)
    return ' '.join(cleaned)


def get_crop(raw):
    if 'Tomato' in raw: return '🍅 Tomato'
    if 'Potato' in raw: return '🥔 Potato'
    if 'Pepper' in raw: return '🫑 Bell Pepper'
    return '🌿 Plant'


# ─── HERO ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌿 AI-Powered Plant Pathology</div>
    <h1>Detect crop<br><span>diseases instantly</span></h1>
    <p>Upload a leaf photo and get instant AI diagnosis with treatment recommendations — powered by deep learning trained on 20,000+ plant images.</p>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-num">97.4%</div>
            <div class="hero-stat-label">Accuracy</div>
        </div>
        <div>
            <div class="hero-stat-num">15</div>
            <div class="hero-stat-label">Disease Classes</div>
        </div>
        <div>
            <div class="hero-stat-num">20K+</div>
            <div class="hero-stat-label">Training Images</div>
        </div>
        <div>
            <div class="hero-stat-num">&lt;1s</div>
            <div class="hero-stat-label">Detection Speed</div>
        </div>
    </div>
</div>
<div class="main-content">
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────
model, class_names, model_loaded = load_model()

if not model_loaded:
    st.error("⚠️ Model file not found. Make sure `crop_disease_model.pth` is in the same folder.")

# ─── Upload + Results ──────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="upload-label">Upload a leaf image</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">Take a clear photo of the affected leaf in natural light for best results</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, use_column_width=True)

        # Tips
        st.markdown("""
        <div class="info-section" style="margin-top:16px">
            <div class="info-section-title">📸 Photo tips for better results</div>
            <div class="info-section-body">
                • Focus on a single leaf with clear symptoms<br>
                • Use natural daylight — avoid flash<br>
                • Fill the frame with the leaf<br>
                • Keep the background simple
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-section" style="text-align:center; padding: 48px 24px;">
            <div style="font-size:48px; margin-bottom:12px">🍃</div>
            <div style="font-family:'Playfair Display',serif; font-size:18px; color:#c8e6d4; margin-bottom:8px">No image uploaded yet</div>
            <div style="font-size:14px; color:#4a7a62">Upload a leaf photo to begin detection</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    if uploaded and model_loaded:
        with st.spinner("🔬 Analyzing leaf..."):
            results = predict(image, model, class_names)

        top_class, top_conf = results[0]
        info = DISEASE_INFO.get(top_class, {
            "severity": "Unknown",
            "color": "#888",
            "symptoms": "No information available.",
            "treatment": "Consult a local agricultural expert.",
            "prevention": "Monitor plant health regularly.",
            "icon": "🔍"
        })

        disease_display = format_name(top_class)
        crop_display = get_crop(top_class)
        severity = info['severity']
        sev_class = f"severity-{severity.lower()}"

        # Main result card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">
                <div class="result-icon">{info['icon']}</div>
                <div>
                    <div class="result-disease">{disease_display}</div>
                    <div class="result-crop">{crop_display}</div>
                </div>
            </div>
            <div class="severity-badge {sev_class}">
                {'🟢' if severity == 'Healthy' else '🟡' if severity == 'Moderate' else '🔴'} {severity} severity
            </div>
            <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:#4a7a62; margin-bottom:10px; font-weight:500;">Top predictions</div>
        """, unsafe_allow_html=True)

        bar_colors = ["#4ecca3", "#2d9e6b", "#1a6b47", "#0f4a30", "#0a3320"]
        for i, (cls, conf) in enumerate(results[:5]):
            name = format_name(cls)
            pct = int(conf * 100)
            color = bar_colors[i]
            st.markdown(f"""
            <div class="conf-row">
                <span class="conf-label">{name}</span>
                <div class="conf-bar-bg"><div class="conf-bar" style="width:{pct}%; background:{color}"></div></div>
                <span class="conf-pct">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Info cards
        st.markdown(f"""
        <div class="info-section">
            <div class="info-section-title">🔬 Symptoms</div>
            <div class="info-section-body">{info['symptoms']}</div>
        </div>
        <div class="info-section">
            <div class="info-section-title">💊 Treatment</div>
            <div class="info-section-body">{info['treatment']}</div>
        </div>
        <div class="info-section">
            <div class="info-section-title">🛡️ Prevention</div>
            <div class="info-section-body">{info['prevention']}</div>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div class="result-card" style="min-height:300px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
            <div style="font-size:52px; margin-bottom:16px">🔬</div>
            <div style="font-family:'Playfair Display',serif; font-size:20px; color:#c8e6d4; margin-bottom:8px">Awaiting analysis</div>
            <div style="font-size:14px; color:#4a7a62; max-width:260px; line-height:1.6">Upload a leaf image on the left to see the AI diagnosis here</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Divider ───────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─── Supported Crops ───────────────────────────────────
st.markdown('<div class="section-title">Supported crops & diseases</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">The model can detect the following conditions across 3 crop types</div>', unsafe_allow_html=True)

crop_col1, crop_col2, crop_col3 = st.columns(3)

with crop_col1:
    st.markdown("""
    <div class="info-section">
        <div class="info-section-title">🍅 Tomato — 10 conditions</div>
        <div class="info-section-body" style="font-size:13px; line-height:2">
            Bacterial Spot<br>Early Blight<br>Late Blight<br>Leaf Mold<br>
            Septoria Leaf Spot<br>Spider Mites<br>Target Spot<br>
            Yellow Leaf Curl Virus<br>Mosaic Virus<br>✅ Healthy
        </div>
    </div>
    """, unsafe_allow_html=True)

with crop_col2:
    st.markdown("""
    <div class="info-section">
        <div class="info-section-title">🥔 Potato — 3 conditions</div>
        <div class="info-section-body" style="font-size:13px; line-height:2">
            Early Blight<br>Late Blight<br>✅ Healthy
        </div>
    </div>
    """, unsafe_allow_html=True)

with crop_col3:
    st.markdown("""
    <div class="info-section">
        <div class="info-section-title">🫑 Bell Pepper — 2 conditions</div>
        <div class="info-section-body" style="font-size:13px; line-height:2">
            Bacterial Spot<br>✅ Healthy
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Stats Row ─────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", "MobileNetV2")
m2.metric("Accuracy", "97.4%")
m3.metric("Training Images", "20,638")
m4.metric("GPU", "RTX 5080")

# ─── Footer ────────────────────────────────────────────
st.markdown("""
</div>
<div class="footer">
    <p>Built with <span>PyTorch + Streamlit</span> · Trained on PlantVillage Dataset · <span>97.4%</span> Validation Accuracy</p>
</div>
""", unsafe_allow_html=True)
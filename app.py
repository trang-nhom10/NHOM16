import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Danh sách lớp phân loại
class_names = ['Benign', 'Malignant', 'Normal']

# Hàm tải mô hình đã huấn luyện
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load("resnet50_fineTuning.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Giao diện Streamlit
st.title("🔬 Dự đoán Ung thư Vú từ Ảnh Siêu Âm (ResNet-50)")

uploaded_file = st.file_uploader("📤 Tải ảnh siêu âm", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

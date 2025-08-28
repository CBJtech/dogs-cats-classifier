import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import gdown
import os

# Model Definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128*12*12, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 128*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device("cpu")  # Streamlit Cloud uses CPU
    model = Net().to(device)
    
    model_path = "model.pth"
    
    # Download model from Google Drive if it doesn't exist locally
    if not os.path.exists(model_path):
        try:
            st.info("📥 Downloading model from Google Drive... This may take a moment.")
            file_id = "1ySr9Xf7p9xoE58ebl0wN-6p2Zu0y09G2"  # file ID
            
            # Create a progress bar for download
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download with gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            
            progress_bar.progress(100)
            status_text.success("✅ Model downloaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            st.error("Please check your Google Drive file ID and sharing permissions.")
            st.stop()
    
    # Load the model
    try:
        # Load model state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please ensure your model was saved correctly.")
        st.stop()

def predict_image(model, uploaded_file, device):
    """Predict uploaded image with EXACT same preprocessing as training"""
    try:
        # Open and process image
        image = Image.open(uploaded_file)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert PIL to numpy array for display
        img_array = np.array(image)
        
        # CRITICAL: Use EXACT same transform pipeline as training
        # Your training transform: transforms.Compose([
        #     transforms.ToTensor(),           # Converts to [0,1] and adds channel dim
        #     transforms.Resize((128, 128)),   # Resize to 128x128
        #     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
        # ])
        
        # Apply identical transforms manually
        # Step 1: Convert to tensor (ToTensor equivalent)
        img_tensor = torch.from_numpy(np.array(image)).float()
        img_tensor = img_tensor / 255.0  # Normalize to [0,1]
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension [H,W] -> [1,H,W]
        
        # Step 2: Resize (must be done on tensor, not numpy array)
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.squeeze(0)  # Remove extra batch dim [1,1,128,128] -> [1,128,128]
        
        # Step 3: Normalize exactly like training
        img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1,1]
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)  # [1,128,128] -> [1,1,128,128]
        
        # Debug: Print tensor stats
        print(f"Tensor shape: {img_tensor.shape}")
        print(f"Tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'image': img_array,
            'prediction': 'Cat' if predicted_class == 0 else 'Dog',
            'confidence': confidence,
            'probabilities': {
                'Cat': probabilities[0][0].item(),
                'Dog': probabilities[0][1].item()
            }
        }
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        print(f"Detailed error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="🐱🐶 Dogs vs Cats Classifier",
        page_icon="🐱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🐱🐶 Dogs vs Cats Classifier")
    st.markdown("Upload an image of a cat or dog and let our AI model classify it!")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write("**Architecture:** Custom CNN")
        st.write("**Input Size:** 128x128 grayscale")
        st.write("**Classes:** Cat (🐱), Dog (🐶)")
        st.write("**Framework:** PyTorch")
        
        st.markdown("---")
        st.header("🔍 How it works")
        st.write("1. Upload a clear image")
        st.write("2. Image is resized to 128x128")
        st.write("3. CNN processes the image")
        st.write("4. Get prediction + confidence")
        
        st.markdown("---")
        st.write("**Tips for best results:**")
        st.write("• Use clear, well-lit images")
        st.write("• Make sure animal is main subject")
        st.write("• Avoid heavily filtered images")
    
    # Load model (this will download if not exists)
    try:
        model, device = load_model()
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of a cat or dog"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None:
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing image..."):
                    result = predict_image(model, uploaded_file, device)
                
                if result is not None:
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    # Confidence styling
                    if confidence > 0.8:
                        confidence_color = "🟢"
                        confidence_text = "High Confidence"
                    elif confidence > 0.6:
                        confidence_color = "🟡"
                        confidence_text = "Medium Confidence"
                    else:
                        confidence_color = "🔴"
                        confidence_text = "Low Confidence"
                    
                    # Main prediction
                    st.success(f"**Prediction: {prediction}** {'🐱' if prediction == 'Cat' else '🐶'}")
                    st.markdown(f"### {confidence_color} Confidence: {confidence:.1%}")
                    st.caption(confidence_text)
                    
                    # Progress bars
                    st.subheader("📊 Class Probabilities")
                    
                    cat_prob = result['probabilities']['Cat']
                    dog_prob = result['probabilities']['Dog']
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write("🐱 **Cat**")
                        st.progress(cat_prob)
                        st.write(f"{cat_prob:.1%}")
                    
                    with col_b:
                        st.write("🐶 **Dog**")
                        st.progress(dog_prob)
                        st.write(f"{dog_prob:.1%}")
                    
                    # Additional metrics
                    st.subheader("📈 Detailed Metrics")
                    
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.metric("Predicted Class", prediction)
                        st.metric("Cat Score", f"{cat_prob:.3f}")
                    with col_d:
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.metric("Dog Score", f"{dog_prob:.3f}")
        
        else:
            st.info("👆 Please upload an image to get started!")
            st.markdown("### 🎯 Try these sample images:")
            st.write("Upload any cat or dog image to test the classifier!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built by <a href="https://cbjtech.github.io/portfolio/"> Cherno Basiru Jallow</a> using PyTorch and Streamlit</p>
            <p>🚀 Deployed on Streamlit Community Cloud</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

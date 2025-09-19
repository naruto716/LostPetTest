# DINOv3 Setup Instructions

## 🔑 **Required: Hugging Face Access Token**

DINOv3 models are gated and require a Hugging Face access token.

### **1. Get Your HF Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Copy the token (starts with `hf_...`)

### **2. Set Up Authentication:**

**Option A: Environment Variable (Recommended)**
```bash
export HF_TOKEN="hf_your_token_here"
```

**Option B: Create .env file**
```bash
# In your project root directory
echo 'HF_TOKEN="hf_your_token_here"' > .env
```

**Option C: Use Hugging Face CLI**
```bash
pip install huggingface_hub
huggingface-cli login
# Enter your token when prompted
```

### **3. Request Access to DINOv3 Models:**
Visit these pages and click "Request Access":
- https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
- https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m

Access is usually granted quickly (within minutes).

### **4. Test Setup:**
```bash
uv run python test_dinov2_vs_dinov3.py
```

If you see authentication errors, double-check your token and access permissions.

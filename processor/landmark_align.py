# processor/landmark_align.py
import numpy as np
import torch
import torch.nn.functional as F

def _to_tensor(img_pil):
    # PIL -> float tensor [0,1], (C,H,W)
    t = torch.from_numpy(np.array(img_pil)).float() / 255.0  # (H,W,3)
    return t.permute(2,0,1).contiguous()

def _to_pil(img_t):
    import numpy as np
    from PIL import Image
    t = (img_t.clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(t)

def simple_align(image_pil, landmarks, out_size=(224,224)):
    """
    最小实现：没有 landmarks -> 仅 resize 到 out_size
    有 landmarks -> 以 landmarks 的外接框做一个粗 ROI，再 resize
    """
    img = _to_tensor(image_pil).unsqueeze(0)  # (1,3,H,W)
    if not landmarks or len(landmarks) < 1:
        return _to_pil(F.interpolate(img, size=out_size, mode="bilinear", align_corners=False).squeeze(0))
    k = np.array(landmarks, dtype=np.float32) # (K,2)
    x1,y1 = np.floor(k[:,0].min()), np.floor(k[:,1].min())
    x2,y2 = np.ceil(k[:,0].max()),  np.ceil(k[:,1].max())
    _,_,H,W = img.shape
    x1,y1 = max(0,int(x1)), max(0,int(y1))
    x2,y2 = min(W-1,int(x2)), min(H-1,int(y2))
    # 裁剪 + resize
    crop = img[:,:, y1:y2+1, x1:x2+1]
    if crop.numel()==0:  # 防御
        crop = img
    crop = F.interpolate(crop, size=out_size, mode="bilinear", align_corners=False)
    return _to_pil(crop.squeeze(0))

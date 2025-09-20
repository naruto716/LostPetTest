#!/usr/bin/env python3
"""
Debug DINOv3 Model Structure
🔍 Investigate the actual model structure to fix hook registration
"""

import torch
from transformers import AutoModel, AutoConfig

def debug_dinov3_structure():
    print("🔍 Investigating DINOv3 Model Structure")
    print("=" * 50)
    
    # Load DINOv3 model
    hf_name = 'facebook/dinov3-vits16-pretrain-lvd1689m'
    model = AutoModel.from_pretrained(hf_name, trust_remote_code=True)
    
    print(f"📋 Model type: {type(model).__name__}")
    print(f"📋 Model class: {model.__class__}")
    
    print("\n🔍 Top-level attributes:")
    for attr in dir(model):
        if not attr.startswith('_') and not callable(getattr(model, attr)):
            try:
                value = getattr(model, attr)
                print(f"   {attr}: {type(value).__name__}")
            except:
                print(f"   {attr}: <error accessing>")
    
    print("\n🔍 Looking for encoder/transformer layers...")
    
    # Check different possible structures
    possible_paths = [
        'encoder.layer',
        'transformer.layers', 
        'blocks',
        'layers',
        'encoder.layers',
        'transformer.h',
        'vit.encoder.layer',
    ]
    
    for path in possible_paths:
        try:
            parts = path.split('.')
            obj = model
            for part in parts:
                obj = getattr(obj, part)
            print(f"✅ Found: {path} -> {len(obj)} layers of type {type(obj[0]).__name__}")
            
            # Show structure of first layer
            print(f"   First layer attributes:")
            for attr in dir(obj[0]):
                if not attr.startswith('_') and not callable(getattr(obj[0], attr)):
                    print(f"      {attr}")
            break
        except AttributeError:
            print(f"❌ Not found: {path}")
    
    print("\n🔍 Full model structure (first 2 levels):")
    def print_structure(obj, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return
        for attr in dir(obj):
            if attr.startswith('_') or callable(getattr(obj, attr, None)):
                continue
            try:
                value = getattr(obj, attr)
                if hasattr(value, '__len__') and not isinstance(value, (str, torch.Tensor)):
                    print(f"{prefix}{attr}: {type(value).__name__} (len={len(value)})")
                    if current_depth < max_depth - 1 and hasattr(value, '__getitem__'):
                        try:
                            print_structure(value[0], prefix + "  ", max_depth, current_depth + 1)
                        except:
                            pass
                else:
                    print(f"{prefix}{attr}: {type(value).__name__}")
            except:
                print(f"{prefix}{attr}: <error>")
    
    print_structure(model)
    
    print("\n🧪 Testing forward pass to see intermediate shapes...")
    dummy_input = torch.randn(1, 3, 224, 224)  # Standard ViT input
    
    try:
        with torch.no_grad():
            outputs = model(pixel_values=dummy_input, output_hidden_states=True)
            print(f"✅ Forward pass successful")
            print(f"   Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
            
            if hasattr(outputs, 'hidden_states'):
                print(f"   Hidden states: {len(outputs.hidden_states)} layers")
                for i, hs in enumerate(outputs.hidden_states):
                    print(f"      Layer {i}: {hs.shape}")
            
            if hasattr(outputs, 'last_hidden_state'):
                print(f"   Last hidden state: {outputs.last_hidden_state.shape}")
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

if __name__ == "__main__":
    debug_dinov3_structure()

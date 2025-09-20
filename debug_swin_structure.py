#!/usr/bin/env python3
"""
Debug SWIN Model Structure
🔍 Find the correct paths for stage extraction
"""

import torch
import timm

def debug_swin_structure():
    print("🔍 Investigating SWIN Model Structure")
    print("=" * 60)
    
    # Load SWIN model
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
    
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
    
    print("\n🔍 Looking for stages/layers...")
    
    # Check different possible paths
    possible_paths = [
        'stages',
        'layers', 
        'features',
        'blocks',
        'encoder',
        'transformer',
    ]
    
    for path in possible_paths:
        try:
            obj = getattr(model, path)
            if hasattr(obj, '__len__'):
                print(f"✅ Found: {path} -> {len(obj)} items of type {type(obj[0]).__name__}")
                
                # Show first stage/layer structure
                print(f"   First item attributes:")
                for attr in dir(obj[0])[:10]:  # Just first 10 to avoid spam
                    if not attr.startswith('_'):
                        print(f"      {attr}")
            else:
                print(f"✅ Found: {path} -> {type(obj).__name__}")
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
    
    print("\n🧪 Testing forward pass to see output shapes...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Forward pass successful")
            print(f"   Final output shape: {output.shape}")
            
            # Try to access intermediate features if available
            if hasattr(model, 'forward_features'):
                features = model.forward_features(dummy_input)
                print(f"   Features shape: {features.shape}")
                
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
    
    print("\n🎯 Manual stage access test:")
    if hasattr(model, 'layers'):
        layers = model.layers
        print(f"   Found {len(layers)} layers/stages")
        
        # Test accessing each stage
        for i, layer in enumerate(layers):
            print(f"   Stage {i}: {type(layer).__name__}")
            
            # Try to hook this stage
            try:
                def test_hook(module, input, output):
                    print(f"      Hook fired! Input: {len(input)} tensors, Output: {type(output)}")
                    if isinstance(output, torch.Tensor):
                        print(f"         Output shape: {output.shape}")
                    elif isinstance(output, tuple):
                        print(f"         Output tuple: {[o.shape if isinstance(o, torch.Tensor) else type(o) for o in output]}")
                
                handle = layer.register_forward_hook(test_hook)
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                handle.remove()
                
            except Exception as e:
                print(f"      ❌ Hook test failed: {e}")

if __name__ == "__main__":
    debug_swin_structure()

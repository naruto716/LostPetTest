"""
Manual inspection script for Dog ReID retrieval results.
Shows ranked gallery images for each query to verify model performance.
"""

import os
import torch
import numpy as np
from datasets import make_dataloaders
from model import make_model
from config_training import cfg
from utils.metrics import euclidean_distance
import logging

def extract_features(model, loader, device):
    """Extract features from a dataloader"""
    model.eval()
    features, pids, paths = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            img, pid, camid, path = batch
            img = img.to(device)
            feat = model(img, 'features')  # L2-normalized features
            
            features.append(feat.cpu())
            pids.extend(pid.numpy())
            paths.extend(path)
    
    features = torch.cat(features, dim=0)
    return features, np.array(pids), paths

def inspect_retrieval_results(model_path=None, top_k=10, num_queries=20):
    """
    Inspect retrieval results by showing top-k gallery matches for queries.
    
    Args:
        model_path: Path to trained model (None for random initialization)
        top_k: Number of top gallery matches to show per query
        num_queries: Number of queries to inspect
    """
    
    print("🔍 Dog ReID Retrieval Inspection")
    print("=" * 50)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print("🏗️  Loading model...")
    model = make_model(
        backbone_name=cfg.BACKBONE,
        num_classes=95,  # Number of training classes
        embed_dim=cfg.EMBED_DIM,
        pretrained=cfg.PRETRAINED,
        bn_neck=cfg.BN_NECK
    )
    
    if model_path and os.path.exists(model_path):
        print(f"📂 Loading trained weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("⚠️  Using random initialization (no trained weights)")
    
    model.to(device)
    model.eval()
    
    # Load data
    print("📊 Loading query and gallery data...")
    _, query_loader, gallery_loader, _ = make_dataloaders(cfg)
    
    # Extract features
    print("🔢 Extracting query features...")
    query_features, query_pids, query_paths = extract_features(model, query_loader, device)
    
    print("🔢 Extracting gallery features...")
    gallery_features, gallery_pids, gallery_paths = extract_features(model, gallery_loader, device)
    
    print(f"Query: {len(query_features)} samples")
    print(f"Gallery: {len(gallery_features)} samples")
    
    # Compute similarity matrix
    print("🧮 Computing similarities...")
    distmat = euclidean_distance(query_features, gallery_features)
    
    # Get rankings for each query
    print("📋 Computing rankings...")
    rankings = np.argsort(distmat, axis=1)  # Sort by distance (ascending)
    
    # Inspect results
    print(f"\n🔍 Inspecting top-{top_k} retrievals for {num_queries} queries:")
    print("=" * 80)
    
    results_summary = []
    
    for q_idx in range(min(num_queries, len(query_features))):
        query_pid = query_pids[q_idx]
        query_path = query_paths[q_idx]
        
        print(f"\n📷 Query {q_idx+1}: {query_path}")
        print(f"   Query PID: {query_pid}")
        
        # Get top-k gallery matches
        top_gallery_indices = rankings[q_idx][:top_k]
        
        correct_retrievals = 0
        gallery_results = []
        
        for rank, g_idx in enumerate(top_gallery_indices):
            gallery_pid = gallery_pids[g_idx]
            gallery_path = gallery_paths[g_idx]
            distance = distmat[q_idx, g_idx]
            is_correct = gallery_pid == query_pid
            
            if is_correct:
                correct_retrievals += 1
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"   Rank {rank+1:2d}: {gallery_path} (PID: {gallery_pid}, dist: {distance:.4f}) {status}")
            
            gallery_results.append({
                'rank': rank + 1,
                'path': gallery_path,
                'pid': gallery_pid,
                'distance': distance,
                'correct': is_correct
            })
        
        precision_at_k = correct_retrievals / top_k
        print(f"   📊 Precision@{top_k}: {precision_at_k:.1%} ({correct_retrievals}/{top_k})")
        
        results_summary.append({
            'query_idx': q_idx,
            'query_path': query_path,
            'query_pid': query_pid,
            'precision_at_k': precision_at_k,
            'correct_retrievals': correct_retrievals,
            'gallery_results': gallery_results
        })
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print("=" * 50)
    
    total_precision = np.mean([r['precision_at_k'] for r in results_summary])
    print(f"Average Precision@{top_k}: {total_precision:.1%}")
    
    # Check for suspiciously perfect results
    perfect_queries = sum(1 for r in results_summary if r['precision_at_k'] == 1.0)
    print(f"Perfect queries (100% precision): {perfect_queries}/{len(results_summary)} ({100*perfect_queries/len(results_summary):.1f}%)")
    
    if perfect_queries / len(results_summary) > 0.8:
        print("🚨 WARNING: >80% perfect queries - possible data leakage or overfitting!")
    
    # Distance analysis
    all_distances = distmat.flatten()
    print(f"\nDistance statistics:")
    print(f"  Min: {all_distances.min():.6f}")
    print(f"  Max: {all_distances.max():.6f}")
    print(f"  Mean: {all_distances.mean():.6f}")
    print(f"  Std: {all_distances.std():.6f}")
    
    near_zero_count = np.sum(all_distances < 0.01)
    print(f"  Near-zero distances (<0.01): {near_zero_count}/{len(all_distances)} ({100*near_zero_count/len(all_distances):.2f}%)")
    
    return results_summary

def save_retrieval_html(results_summary, output_file="retrieval_results.html"):
    """Save results as HTML for easy visual inspection"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dog ReID Retrieval Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .query { border: 2px solid #333; margin: 20px 0; padding: 15px; }
            .gallery { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
            .gallery-item { border: 1px solid #ccc; padding: 5px; width: 200px; }
            .correct { border-color: green; background-color: #e8f5e8; }
            .wrong { border-color: red; background-color: #fce8e8; }
            .image-path { font-family: monospace; font-size: 12px; word-break: break-all; }
        </style>
    </head>
    <body>
        <h1>🔍 Dog ReID Retrieval Results</h1>
    """
    
    for result in results_summary:
        html_content += f"""
        <div class="query">
            <h3>Query {result['query_idx']+1}: PID {result['query_pid']}</h3>
            <div class="image-path">{result['query_path']}</div>
            <p><strong>Precision@{len(result['gallery_results'])}: {result['precision_at_k']:.1%}</strong></p>
            
            <h4>Top Retrieved Gallery Images:</h4>
            <div class="gallery">
        """
        
        for gallery_result in result['gallery_results']:
            css_class = "correct" if gallery_result['correct'] else "wrong"
            status = "✅ CORRECT" if gallery_result['correct'] else "❌ WRONG"
            
            html_content += f"""
                <div class="gallery-item {css_class}">
                    <strong>Rank {gallery_result['rank']}</strong><br>
                    PID: {gallery_result['pid']}<br>
                    Distance: {gallery_result['distance']:.4f}<br>
                    <div class="image-path">{gallery_result['path']}</div>
                    <div>{status}</div>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"💾 Saved detailed results to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect Dog ReID retrieval results")
    parser.add_argument("--model", type=str, default="outputs/dogreid_training/best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top retrievals to show per query")
    parser.add_argument("--num_queries", type=int, default=20,
                       help="Number of queries to inspect")
    parser.add_argument("--save_html", action="store_true",
                       help="Save results as HTML file")
    
    args = parser.parse_args()
    
    # Run inspection
    results = inspect_retrieval_results(
        model_path=args.model,
        top_k=args.top_k, 
        num_queries=args.num_queries
    )
    
    # Save HTML if requested
    if args.save_html:
        save_retrieval_html(results)

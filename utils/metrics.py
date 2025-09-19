"""
Evaluation metrics for Dog ReID.
Based on CLIP-ReID's proven implementation.
"""

import torch
import numpy as np

def euclidean_distance(qf, gf):
    """
    Compute euclidean distance matrix between query and gallery features.
    
    Args:
        qf: Query features with shape (m, d)
        gf: Gallery features with shape (n, d)
    Returns:
        dist_mat: Distance matrix with shape (m, n)
    """
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Evaluation with market1501 metric.
    Key: for each query identity, its gallery images from the same camera view are discarded.
    
    Args:
        distmat: Distance matrix with shape (num_query, num_gallery)
        q_pids: Query person IDs
        g_pids: Gallery person IDs  
        q_camids: Query camera IDs
        g_camids: Gallery camera IDs
        max_rank: Maximum rank to consider
    Returns:
        cmc: Cumulative matching characteristics
        mAP: Mean average precision
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # Compute CMC curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    
    for q_idx in range(num_q):
        # Get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # Compute CMC curve
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # This condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # Compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

class R1_mAP_eval():
    """
    Rank-1 and mean Average Precision evaluation for ReID.
    Following CLIP-ReID's exact implementation.
    """
    
    def __init__(self, num_query, max_rank=50, feat_norm=True):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        """Reset the evaluator state."""
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        """
        Update evaluator with batch output.
        
        Args:
            output: Tuple of (feat, pid, camid)
        """
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        """
        Compute final CMC and mAP scores.
        
        Returns:
            cmc: Cumulative matching characteristics
            mAP: Mean average precision
        """
        feats = torch.cat(self.feats, dim=0)
        
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        
        # Split query and gallery
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        
        print('=> Computing DistMat with euclidean_distance')
        distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank)
        
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

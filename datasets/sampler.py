import random
from collections import defaultdict
from torch.utils.data import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample P identities, with K instances each, to form a batch (P×K).
    This ensures each batch contains multiple images from the same identities,
    which is crucial for triplet loss and other metric learning approaches.
    
    Args:
        data_source: Dataset that returns (img, pid, camid, path) tuples
        batch_size: Total batch size (must be divisible by num_instances)
        num_instances: Number of instances (K) per identity in each batch
    
    Example:
        If batch_size=64 and num_instances=4, then we sample P=16 identities
        with K=4 instances each per batch.
    """
    
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances  # K
        
        # Build index mapping: PID -> list of sample indices
        # Access PIDs directly from dataset (avoid loading images!)
        if hasattr(data_source, 'pids'):
            # Fast path: dataset has PIDs as attribute
            self.index_pid = data_source.pids
        else:
            # Fallback: iterate through dataset (slow, loads images)
            self.index_pid = [pid for _, pid, _, _ in data_source]
        
        self.pid_index = defaultdict(list)
        
        for index, pid in enumerate(self.index_pid):
            self.pid_index[pid].append(index)
            
        self.pids = list(self.pid_index.keys())
        
        # Validate batch configuration
        assert batch_size % num_instances == 0, \
            f"batch_size ({batch_size}) must be divisible by num_instances ({num_instances})"
            
        self.num_pids_per_batch = batch_size // num_instances  # P
        
        print(f"RandomIdentitySampler: {len(self.pids)} identities, "
              f"P={self.num_pids_per_batch} ids per batch, K={self.num_instances} instances per id")

    def __iter__(self):
        """
        Generate batches by sampling P identities and K instances each.
        For identities with fewer than K instances, we oversample with replacement.
        """
        # Prepare batches for each identity
        batch_idxs_dict = defaultdict(list)
        
        for pid in self.pids:
            idxs = self.pid_index[pid].copy()
            
            # If we have fewer instances than needed, oversample with replacement
            if len(idxs) < self.num_instances:
                # Repeat existing samples to reach num_instances
                additional_samples = random.choices(idxs, k=self.num_instances - len(idxs))
                idxs.extend(additional_samples)
                
            # Shuffle the indices
            random.shuffle(idxs)
            
            # Create batches of size num_instances
            batch_idxs = []
            for i in range(0, len(idxs), self.num_instances):
                batch = idxs[i:i + self.num_instances]
                if len(batch) == self.num_instances:
                    batch_idxs.append(batch)
                    
            batch_idxs_dict[pid] = batch_idxs

        # Sample batches until we run out of identities
        available_pids = self.pids.copy()
        final_idxs = []
        
        while len(available_pids) >= self.num_pids_per_batch:
            # Sample P identities for this batch
            selected_pids = random.sample(available_pids, self.num_pids_per_batch)
            
            # Add K instances from each selected identity
            for pid in selected_pids:
                if batch_idxs_dict[pid]:
                    # Pop one batch of K instances for this identity
                    batch_instances = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_instances)
                    
                    # Remove identity if no more batches available
                    if len(batch_idxs_dict[pid]) == 0:
                        available_pids.remove(pid)
                        
        return iter(final_idxs)

    def __len__(self):
        """
        Return the total number of samples that will be generated.        
        The sampler creates batches by:
        1. For each identity, create K-sized mini-batches
        2. Sample P identities and take one mini-batch from each
        3. Continue until not enough identities left
        
        Total batches = (total mini-batches across all identities) / P
        """
        # Count total mini-batches across all identities
        total_mini_batches = 0
        for pid in self.pids:
            num_images = len(self.pid_index[pid])
            # Each identity contributes floor(num_images / K) mini-batches
            num_mini_batches = num_images // self.num_instances
            if num_mini_batches > 0:
                total_mini_batches += num_mini_batches
        
        # Total batches = mini-batches / P
        # (Each final batch needs P mini-batches, one from each selected identity)
        num_batches = total_mini_batches // self.num_pids_per_batch
        
        # Return total samples
        return num_batches * self.batch_size

import csv
from pathlib import Path

def check_pid_overlap(train_csv, test_query_csv):
    """Check for overlapping PIDs between train and test query sets."""
    
    # Read train PIDs
    train_pids = set()
    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_pids.add(row['pid'])
    
    # Read test query PIDs
    test_query_pids = set()
    with open(test_query_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_query_pids.add(row['pid'])
    
    # Find overlaps
    overlapping_pids = train_pids.intersection(test_query_pids)
    
    # Print results
    print("=" * 70)
    print("PID OVERLAP ANALYSIS")
    print("=" * 70)
    print(f"\nTrain set ({train_csv}):")
    print(f"  - Total unique PIDs: {len(train_pids)}")
    
    print(f"\nTest Query set ({test_query_csv}):")
    print(f"  - Total unique PIDs: {len(test_query_pids)}")
    
    print(f"\nOverlapping PIDs: {len(overlapping_pids)}")
    
    if overlapping_pids:
        print(f"\n⚠️  WARNING: Found {len(overlapping_pids)} overlapping PIDs!")
        print("\nFirst 20 overlapping PIDs:")
        for pid in sorted(overlapping_pids)[:20]:
            print(f"  - {pid}")
        
        if len(overlapping_pids) > 20:
            print(f"  ... and {len(overlapping_pids) - 20} more")
        
        # Show some examples from train
        print("\nExamples in train set:")
        with open(train_csv, 'r') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if row['pid'] in overlapping_pids and count < 5:
                    print(f"  - {row['img_rel_path']} (pid={row['pid']})")
                    count += 1
        
        # Show some examples from test query
        print("\nExamples in test query set:")
        with open(test_query_csv, 'r') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if row['pid'] in overlapping_pids and count < 5:
                    print(f"  - {row['img_rel_path']} (pid={row['pid']})")
                    count += 1
    else:
        print("\n✅ No overlapping PIDs found - data splits are properly isolated!")
    
    print("\n" + "=" * 70)
    
    return overlapping_pids

if __name__ == "__main__":
    train_csv = "splits_petface_valid/train.csv"
    test_query_csv = "splits_petface_test_10k/test_query.csv"
    test_gallery_csv = "splits_petface_test_10k/test_gallery.csv"
    
    # Check train vs test_query
    overlaps = check_pid_overlap(train_csv, test_query_csv)
    
    # Check train vs test_gallery
    print("\n")
    overlaps_gallery = check_pid_overlap(train_csv, test_gallery_csv)
    
    # Save overlapping PIDs to file if found
    if overlaps or overlaps_gallery:
        with open("overlapping_pids.txt", 'w') as f:
            for pid in sorted(overlaps.union(overlaps_gallery)):
                f.write(f"{pid}\n")
        print(f"\nOverlapping PIDs saved to: overlapping_pids.txt")


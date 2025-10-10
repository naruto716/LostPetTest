"""
Sample a subset of identities from the petface re-id splits for faster modelling.
"""

import pandas as pd
import numpy as np
import argparse

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def sample_identities(input_csv, output_csv, n_identities):
    df = pd.read_csv(input_csv)
    # Always use 'pid' as the identity column for petface splits
    if 'pid' not in df.columns:
        raise ValueError(f"Expected 'pid' column in {input_csv}, but columns are: {df.columns}")
    id_col = 'pid'
    unique_ids = df[id_col].unique()
    if len(unique_ids) < n_identities:
        raise ValueError(f"Requested {n_identities} identities, but only {len(unique_ids)} available in {input_csv}")
    sampled_ids = np.random.choice(unique_ids, n_identities, replace=False)
    subset_df = df[df[id_col].isin(sampled_ids)]
    subset_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(subset_df)} rows for {n_identities} identities to {output_csv}")
    return subset_df, id_col


def sample_test_query_gallery(test_query_csv, test_gallery_csv, query_out, gallery_out, n_test):
    # For test, sample identities from test_query.csv, then filter both test_query and test_gallery
    test_query_df = pd.read_csv(test_query_csv)
    if 'pid' not in test_query_df.columns:
        raise ValueError(f"Expected 'pid' column in {test_query_csv}, but columns are: {test_query_df.columns}")
    id_col = 'pid'
    unique_ids = test_query_df[id_col].unique()
    if len(unique_ids) < n_test:
        raise ValueError(f"Requested {n_test} identities, but only {len(unique_ids)} available in {test_query_csv}")
    sampled_ids = np.random.choice(unique_ids, n_test, replace=False)
    # Filter test_query and test_gallery for these pids
    test_query_subset = test_query_df[test_query_df[id_col].isin(sampled_ids)]
    test_query_subset.to_csv(query_out, index=False)
    print(f"Wrote {len(test_query_subset)} rows for {n_test} identities to {query_out}")

    test_gallery_df = pd.read_csv(test_gallery_csv)
    if 'pid' not in test_gallery_df.columns:
        raise ValueError(f"Expected 'pid' column in {test_gallery_csv}, but columns are: {test_gallery_df.columns}")
    test_gallery_subset = test_gallery_df[test_gallery_df[id_col].isin(sampled_ids)]
    test_gallery_subset.to_csv(gallery_out, index=False)
    print(f"Wrote {len(test_gallery_subset)} rows for {n_test} identities to {gallery_out}")


def main():
    parser = argparse.ArgumentParser(description="Sample identities from re-id split CSVs.")
    parser.add_argument('--train_csv', type=str, default='LostPetTest/splits_petface/train.csv')
    parser.add_argument('--test_query_csv', type=str, default='LostPetTest/splits_petface/test_query.csv')
    parser.add_argument('--test_gallery_csv', type=str, default='LostPetTest/splits_petface/test_gallery.csv')
    parser.add_argument('--train_out', type=str, default='subset_splits_petface/train_subset.csv')
    parser.add_argument('--query_out', type=str, default='subset_splits_petface/test_query_subset.csv')
    parser.add_argument('--gallery_out', type=str, default='subset_splits_petface/test_gallery_subset.csv')
    parser.add_argument('--n_train', type=int, default=400)
    parser.add_argument('--n_test', type=int, default=100)
    args = parser.parse_args()

    # Train subset as before
    sample_identities(args.train_csv, args.train_out, args.n_train)

    sample_test_query_gallery(
        args.test_query_csv,
        args.test_gallery_csv,
        args.query_out,
        args.gallery_out,
        args.n_test
    )

if __name__ == "__main__":
    main()

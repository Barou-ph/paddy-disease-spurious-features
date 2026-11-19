"""Build the project's normalized manifest from Kaggle Paddy Disease data.

Expected input layout:
    data/raw/train.csv
    data/raw/<class_name>/<image_id>

Run from the repository root:
    python scripts/create_manifest.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/raw/train.csv')
    parser.add_argument('--image-root', default='data/raw')
    parser.add_argument('--output', default='data/processed/manifest.csv')
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    image_root = ROOT / args.image_root
    output_path = ROOT / args.output

    if not csv_path.is_file():
        raise FileNotFoundError(f'Cannot find labels CSV: {csv_path}')
    if not image_root.is_dir():
        raise FileNotFoundError(f'Cannot find image root: {image_root}')

    source = pd.read_csv(csv_path)
    required = {'image_id', 'label', 'variety', 'age'}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f'train.csv is missing columns: {sorted(missing)}')
    if source.empty:
        raise ValueError('train.csv contains no rows.')
    if source['image_id'].isna().any() or source['label'].isna().any():
        raise ValueError('train.csv contains missing image_id or label values.')
    if source['image_id'].duplicated().any():
        examples = source.loc[source['image_id'].duplicated(), 'image_id'].head().tolist()
        raise ValueError(f'Duplicate image_id values in train.csv, e.g. {examples}')

    rows = []
    missing_images = []
    label_folder_mismatches = []
    for row in source.itertuples(index=False):
        image_id = str(row.image_id)
        label = str(row.label)
        image_path = image_root / label / image_id
        if not image_path.is_file():
            missing_images.append(str(image_path.relative_to(ROOT)))
            continue
        rows.append({
            # Relative to image_root (data/raw), matching baseline.yaml.
            'image_path': f'{label}/{image_id}',
            'label': label,
            'variety': row.variety,
            'age': row.age,
        })

    if missing_images:
        preview = '\n'.join(missing_images[:20])
        raise FileNotFoundError(
            f'{len(missing_images)} image(s) listed in train.csv were not found.\n'
            f'First missing paths:\n{preview}'
        )

    manifest = pd.DataFrame(rows)
    # Validate that every expected class has samples and that there are exactly 10.
    class_counts = manifest['label'].value_counts().sort_index()
    if len(class_counts) != 10:
        raise ValueError(f'Expected 10 classes, found {len(class_counts)}: {class_counts.to_dict()}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    print(f'Manifest created: {output_path.relative_to(ROOT)}')
    print(f'Rows: {len(manifest):,}')
    print(f'Columns: {manifest.columns.tolist()}')
    print('\nClass distribution:')
    print(class_counts.to_string())
    print('\nMetadata missing values:')
    print(manifest[['variety', 'age']].isna().sum().to_string())
    print(f'Unique image_id: {source.image_id.nunique():,}')
    print(f'Image files found: {len(manifest):,}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise

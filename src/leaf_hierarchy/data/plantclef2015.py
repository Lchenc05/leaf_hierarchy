"""Inventory, audit, filter and split PlantCLEF2015 tree leaf scans.

The split reproduces the original experiment: tree decisions are frozen in
taxonomy_snapshot.json, and image paths use a canonical dataset prefix so that
group identifiers do not depend on the location of the downloaded data.
Generated manifests use a shared English schema; the historical split protocol is preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
from importlib import metadata
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import platform
import random
import re
import shutil
from uuid import uuid4
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn

from leaf_hierarchy.config import load_data_config
from leaf_hierarchy.runtime import ROOT
from .schema import english_artifact_frame
from .manifest import MANIFEST_COLUMNS, legacy_split_fingerprint, split_fingerprint
from .taxonomy import build_taxonomy, taxonomy_fingerprint
DATASET_PREFIX = "PlantCLEF2015TrainingData"
SEED = 20260909
SPLIT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}
MIN_GROUPS = 20
MIN_EVAL_GROUPS = 3
MIN_TRAIN_GROUPS = 14
PHASH_DISTANCE = 4
VISUAL_IOU_MIN = 0.95
VISUAL_FOREGROUND_MAE_MAX = 0.045
VISUAL_SIZE = 64


def canonical_path(path: Path, data_dir: Path) -> str:
    """Preserve manifest paths used to generate the original stable group IDs."""
    return (PurePosixPath(DATASET_PREFIX) / path.relative_to(data_dir).as_posix()).as_posix()


def resolve_image_path(image_path: str, data_dir: Path) -> Path:
    """Resolve a canonical manifest entry beneath the supplied dataset directory."""
    relative = PurePosixPath(image_path)
    if relative.is_absolute() or not relative.parts or relative.parts[0] != DATASET_PREFIX or ".." in relative.parts:
        raise ValueError(f"Invalid canonical dataset path: {image_path}")
    resolved = data_dir.joinpath(*relative.parts[1:]).resolve()
    if not resolved.is_relative_to(data_dir.resolve()):
        raise ValueError(f"Image path escapes the dataset directory: {image_path}")
    return resolved


def validate_paths(data_dir: Path, output_dir: Path, taxonomy_file: Path) -> None:
    """Fail before writing if inputs are unavailable or outputs overlap them."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}. Set [data].data_dir in --config, "
            "LEAF_HIERARCHY_DATA_DIR or --data-dir to point to PlantCLEF2015TrainingData."
        )
    if not taxonomy_file.is_file():
        raise FileNotFoundError(f"Botanical source snapshot does not exist: {taxonomy_file}")
    if output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError("The output directory must not overlap the original dataset directory.")
    if output_dir == ROOT:
        raise ValueError("Choose a separate output directory instead of the project root.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}. Choose a new --output-dir.")


def build_config(workers: int, data_dir: Path, taxonomy_file: Path) -> dict:
    """Record the fixed experiment settings and the installed analysis packages."""
    versions = {"python": platform.python_version(), **{
        package: metadata.version(package)
        for package in ["numpy", "pandas", "Pillow", "scipy", "matplotlib"]
    }}
    return {
        "seed": SEED,
        "split_ratios_groups": SPLIT_RATIOS,
        "min_groups_per_species": MIN_GROUPS,
        "min_groups_validation_test": MIN_EVAL_GROUPS,
        "min_groups_train": MIN_TRAIN_GROUPS,
        "phash_hamming_threshold": PHASH_DISTANCE,
        "phash_bits": 64,
        "phash_active_bits": 63,
        "phash_orientations": 8,
        "image_workers": workers,
        "visual_iou_min": VISUAL_IOU_MIN,
        "visual_foreground_mae_max": VISUAL_FOREGROUND_MAE_MAX,
        "visual_thumbnail_size": VISUAL_SIZE,
        "visual_max_translation_pixels": 1,
        "csv_encoding": "utf-8",
        "csv_separator": ",",
        "csv_missing_value": "",
        "taxonomy_source_date": "2026-09-09",
        "status": "grouped_species_split",
        "manifest_schema_version": 1,
        "protocol": "plantclef2015_tree_leaf_scans_v1",
        "taxonomy_labels_source": "dataset_labels",
        "versions": versions,
        "data_dir": str(data_dir),
        "canonical_dataset_prefix": DATASET_PREFIX,
        "taxonomy_file": str(taxonomy_file),
        "taxonomy_snapshot_sha256": hashlib.sha256(taxonomy_file.read_bytes()).hexdigest(),
    }


def write_csv(frame, name, *, output_dir, config):
    """Export the English artifact schema without a DataFrame index."""
    english_artifact_frame(frame).to_csv(output_dir / name, index=False, encoding=config["csv_encoding"],
                 sep=config["csv_separator"], na_rep=config["csv_missing_value"])

def load_taxonomic_criteria(taxonomy_file, leafscan_species_labels):

    """
    Load saved botanical decisions and verify their supporting source records.
    Reproduction uses the preserved source extracts in taxonomy_snapshot.json.
    ``leafscan_species_labels`` contains Species labels from the original XML files.
    """

    # Load the frozen botanical decisions and their supporting source records.
    snapshot = json.loads(Path(taxonomy_file).read_text(encoding='utf-8'))
    criteria = pd.DataFrame(snapshot['criterios'])
    criteria['decision'] = criteria['decision'].replace({'incluir': 'include', 'excluir': 'exclude', 'pendiente': 'pending'})
    sources = pd.DataFrame(snapshot['extractos'])
    provenance = pd.DataFrame(snapshot['procedencia'])
    assert criteria['Species'].is_unique, 'A species label has multiple botanical decisions.'
    assert set(criteria['Species']) == set(leafscan_species_labels), 'Review the botanical table: the set of LeafScan species has changed.'
    assert set(criteria['decision']) <= {'include', 'exclude', 'pending'}
    assert not sources.duplicated(['source_dataset', 'source_record_id']).any()
    assert set(provenance['source_dataset']) == {'BGCI', 'WCVP'}
    refs = {(r.source_dataset, r.source_record_id): r for r in sources.itertuples(index=False)}

    def normalize_taxon_label(label):
        tokens = label.split()
        if len(tokens) > 2 and tokens[1] in {'x', '×'}:
            tokens.pop(1)
        return re.sub('[.\\s]', '', ' '.join(tokens)).casefold()

    # Check that each decision has a matching source and scientific name.
    for row in criteria.itertuples(index=False):
        assert row.source_url and row.source_title and row.consulted_on, row.Species
        assert (row.source_dataset, row.source_taxon_id) in refs, f'Missing source snapshot: {row.Species}'
        source = refs[row.source_dataset, row.source_taxon_id]
        if row.decision in {'include', 'exclude'}:
            assert normalize_taxon_label(row.Species) == normalize_taxon_label(source.full_name), f'Scientific name or author does not match: {row.Species}'
        if row.decision == 'include':
            assert row.evidence and row.accepted_name, row.Species
            if row.source_dataset == 'BGCI':
                assert source.taxon_status == 'accepted', row.Species
            else:
                accepted = refs['WCVP', source.accepted_record_id]
                assert accepted.taxon_status in {'Accepted', 'Artificial Hybrid'}, row.Species
                assert 'tree' in accepted.lifeform_description, row.Species
        if row.wcvp_name_id:
            assert ('WCVP', row.wcvp_name_id) in refs, row.Species
    return (criteria, sources, provenance)

# Count images and distinct observations for each species.
def species_summary(frame):
    base = frame.groupby('Species').agg(**{'images': ('image_path', 'size'), 'genus': ('Genus', 'first'), 'family': ('Family', 'first')})
    obs = frame.loc[frame.ObservationId.ne('')].groupby('Species').ObservationId.nunique()
    base['observations'] = obs.reindex(base.index, fill_value=0).astype(int)
    base['images_without_observation'] = frame.ObservationId.eq('').groupby(frame.Species).sum().astype(int)
    base['images_per_observation'] = base["images"] / base["observations"].replace(0, np.nan)
    return base.reset_index().sort_values(['observations', 'images', 'Species'], ascending=[False, False, True])


# Keep related images together in a single partition.
class UnionFind:

    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = (self.find(a), self.find(b))
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = (b, a)
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

class BKTree:

    """Exact Hamming-distance search without approximate hash neighbors."""

    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, threshold):
        stack = [self.root] if self.root else []
        hits = []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                hits.extend(((i, distance) for i in node[1]))
            stack.extend((child for d, child in node[2].items() if distance - threshold <= d <= distance + threshold))
        return hits

def inventory_dataset(data_dir, output_dir, config):
    """Read original XML/JPG associations and export the complete metadata audit."""
    save_csv = partial(write_csv, output_dir=output_dir, config=config)
    def project_relative(path):
        return canonical_path(path, data_dir)

    # Find XML metadata and image files in the original dataset.
    xml_files = sorted((p for p in data_dir.rglob('*') if p.is_file() and p.suffix.lower() == '.xml'))
    jpg_files = sorted((p for p in data_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg'}))
    if not xml_files:
        raise ValueError('No XML files found in PlantCLEF2015TrainingData')
    xml_samples = []
    for xml_path in xml_files[:3]:
        xml_samples.append({'xml_path': project_relative(xml_path), 'xml_text': xml_path.read_text(encoding='utf-8', errors='replace')})
    FIELDS = ['ObservationId', 'Species', 'Genus', 'Family', 'Content', 'ClassId', 'MediaId', 'Author', 'Date', 'Location', 'Latitude', 'Longitude', 'YearInCLEF', 'ObservationId2014', 'ImageId2014', 'LearnTag', 'Vote']
    CORE_FIELDS = ['ObservationId', 'Species', 'Genus', 'Family', 'Content', 'ClassId', 'MediaId']

    # Match images to XML files with the same stem and parent directory.
    jpg_by_stem = defaultdict(list)
    for image_path in jpg_files:
        jpg_by_stem[image_path.parent.as_posix().casefold(), image_path.stem.casefold()].append(image_path)
    records_by_image = defaultdict(list)
    xml_errors, anomalies = ([], [])
    structure = Counter()
    structure_examples = {}
    all_tag_counts = Counter()

    def note(image_path, xml_path, kind, field='', value='', detail=''):
        anomalies.append({'image_path': image_path, 'xml_path': xml_path, 'type': kind, 'field': field, 'value': value, 'detail': detail})

    def parse_xml_readonly(path):
        try:
            return (path, ET.parse(path).getroot(), None)
        except (ET.ParseError, OSError, UnicodeError) as exc:
            return (path, None, exc)

    # Read XML files concurrently in bounded batches.
    def iter_parsed_xml(paths):
        with ThreadPoolExecutor(max_workers=24) as pool:
            for offset in range(0, len(paths), 2000):
                yield from pool.map(parse_xml_readonly, paths[offset:offset + 2000])
                if (offset + 2000) % 10000 == 0 or offset + 2000 >= len(paths):
                    print(f'  Read {min(offset + 2000, len(paths)):,}/{len(paths):,} XML files.', flush=True)

    # Extract metadata and record missing, ambiguous or unexpected fields.
    for xml_path, xml_root, parse_error in iter_parsed_xml(xml_files):
        xml_rel = project_relative(xml_path)
        candidates = jpg_by_stem.get((xml_path.parent.as_posix().casefold(), xml_path.stem.casefold()), [])
        paired_paths = candidates or [xml_path.with_suffix('.jpg')]
        values = {field: '' for field in FIELDS}
        raw_values = {field + '_raw': '' for field in FIELDS}
        parse_status = 'valid'
        local_notes = []
        try:
            if parse_error is not None:
                raise parse_error

            def walk_structure(element, parent=''):
                here = f'{parent}/{element.tag}'
                structure[here] += 1
                structure_examples.setdefault(here, xml_rel)
                for child in element:
                    walk_structure(child, here)
            walk_structure(xml_root)
            children = defaultdict(list)
            for child in xml_root:
                children[child.tag].append(child)
                all_tag_counts[child.tag] += 1
            if xml_root.tag != 'Image':
                parse_status = 'unexpected_structure'
                local_notes.append(('unexpected_xml_root', 'root', xml_root.tag, 'Review required; this record will not be selected automatically.'))
            for field in FIELDS:
                nodes = children.get(field, [])
                if len(nodes) > 1:
                    parse_status = 'unexpected_structure'
                    local_notes.append(('repeated_tag', field, str(len(nodes)), 'The canonical value is left empty because it is ambiguous.'))
                    raw_values[field + '_raw'] = json.dumps([n.text or '' for n in nodes], ensure_ascii=False)
                elif nodes:
                    if len(nodes[0]):
                        parse_status = 'unexpected_structure'
                        local_notes.append(('nested_metadata', field, ET.tostring(nodes[0], encoding='unicode'), 'Review required.'))
                    else:
                        raw = nodes[0].text or ''
                        raw_values[field + '_raw'] = raw
                        values[field] = raw.strip()
            for unknown in set(children) - set(FIELDS):
                local_notes.append(('additional_tag', unknown, '', 'Listed in xml_structure.csv; do not infer its meaning.'))
            if values['MediaId'] and values['MediaId'] != xml_path.stem:
                local_notes.append(('media_id_filename_mismatch', 'MediaId', values['MediaId'], f'XML filename: {xml_path.stem}; association retained using matching sibling filenames.'))
        except (ET.ParseError, OSError, UnicodeError) as exc:
            parse_status = 'read_error'
            xml_errors.append({'xml_path': xml_rel, 'error_type': type(exc).__name__, 'error': str(exc)})
        for image_path in paired_paths:
            image_rel = project_relative(image_path)
            row = {'image_path': image_rel, 'xml_path': xml_rel, 'xml_status': parse_status, **values, **raw_values}
            records_by_image[image_rel].append(row)
            for kind, field, value, detail in local_notes:
                note(image_rel, xml_rel, kind, field, value, detail)
            if len(candidates) > 1:
                note(image_rel, xml_rel, 'multiple_jpg_per_xml', detail=json.dumps([project_relative(p) for p in candidates]))

    # Create one row per image path, including unpaired files.
    actual_jpg = {project_relative(p) for p in jpg_files}
    inventory_rows = []
    for record_number, image_rel in enumerate(sorted(actual_jpg | set(records_by_image)), start=1):
        metadata_rows = records_by_image.get(image_rel, [])
        exists = image_rel in actual_jpg
        if not metadata_rows:
            row = {'image_path': image_rel, 'xml_path': '', 'xml_status': 'missing_xml', **{f: '' for f in FIELDS}, **{f + '_raw': '' for f in FIELDS}}
        elif len(metadata_rows) == 1:
            row = metadata_rows[0].copy()
        else:
            row = metadata_rows[0].copy()
            row['xml_path'] = json.dumps([r['xml_path'] for r in metadata_rows], ensure_ascii=False)
            row['xml_status'] = 'multiple_xml'
            for field in FIELDS:
                options = {r[field] for r in metadata_rows}
                row[field] = next(iter(options)) if len(options) == 1 else ''
                row[field + '_raw'] = json.dumps([r[field + '_raw'] for r in metadata_rows], ensure_ascii=False)
            note(image_rel, row['xml_path'], 'multiple_xml_per_image', detail='This conflict is not resolved automatically.')
        row['image_exists'] = exists
        row['xml_count'] = len(metadata_rows)
        row['source_record'] = record_number
        row['record_type'] = 'image_and_xml' if exists and metadata_rows else 'jpg_without_xml' if exists else 'xml_without_jpg'
        if row['record_type'] != 'image_and_xml':
            note(image_rel, row['xml_path'], row['record_type'])
        for field in FIELDS:
            if not row[field]:
                note(image_rel, row['xml_path'], 'missing_metadata', field, detail='Core inventory field' if field in CORE_FIELDS else 'Optional field; left empty')
        inventory_rows.append(row)
    inventory = pd.DataFrame(inventory_rows)
    first_columns = ['source_record', 'record_type', 'image_path', 'xml_path', *FIELDS, 'image_exists', 'xml_status', 'xml_count']
    inventory = inventory[first_columns + [f + '_raw' for f in FIELDS]]
    assert inventory['image_path'].is_unique, 'The inventory must contain one row per image path.'
    assert inventory['source_record'].is_unique
    assert set(inventory.loc[inventory.image_exists, 'image_path']) == actual_jpg
    anomaly_frame = pd.DataFrame(anomalies, columns=['image_path', 'xml_path', 'type', 'field', 'value', 'detail'])
    anomaly_frame.insert(0, 'source_record', anomaly_frame.image_path.map(inventory.set_index('image_path')["source_record"]))

    # Detect contradictory relationships between labels and observations.
    taxonomic_issues = []
    for source_field, target_field in [('Species', 'Genus'), ('Species', 'Family'), ('Species', 'ClassId'), ('ClassId', 'Species'), ('Genus', 'Family'), ('ObservationId', 'Species'), ('ObservationId', 'ClassId')]:
        usable = inventory[(inventory[source_field] != '') & (inventory[target_field] != '')]
        for key, group in usable.groupby(source_field, sort=True):
            alternatives = sorted(group[target_field].unique())
            if len(alternatives) > 1:
                taxonomic_issues.append({'key_field': source_field, 'key': key, 'conflicting_field': target_field, 'values': json.dumps(alternatives, ensure_ascii=False), 'n_images': len(group)})
    genus_mismatch = inventory[(inventory.Species != '') & (inventory.Genus != '') & (inventory.Species.str.split().str[0] != inventory.Genus)]
    for species, group in genus_mismatch.groupby('Species', sort=True):
        taxonomic_issues.append({'key_field': 'Species', 'key': species, 'conflicting_field': 'genus_vs_species_first_token', 'values': json.dumps(sorted(group.Genus.unique()), ensure_ascii=False), 'n_images': len(group)})
    observation_consistency = inventory[inventory.ObservationId.ne('')].groupby('ObservationId', sort=True).agg(**{'n_images': ('image_path', 'size'), 'n_species': ('Species', 'nunique'), 'n_classes': ('ClassId', 'nunique'), 'n_authors': ('Author', 'nunique'), 'n_dates': ('Date', 'nunique'), 'n_editions': ('YearInCLEF', 'nunique'), 'n_observation_ids_2014': ('ObservationId2014', lambda x: x[x.ne('')].nunique()), 'n_missing_observation_ids_2014': ('ObservationId2014', lambda x: int(x.eq('').sum()))}).reset_index()
    profile_rows = [{'metric': 'actual_jpg_files', 'value': len(jpg_files)}, {'metric': 'actual_xml_files', 'value': len(xml_files)}, {'metric': 'inventory_rows', 'value': len(inventory)}, {'metric': 'xml_errors', 'value': len(xml_errors)}, {'metric': 'taxonomic_inconsistencies', 'value': len(taxonomic_issues)}]
    for field in ['record_type', 'xml_status', 'Content', 'YearInCLEF', 'LearnTag']:
        for value, count in inventory[field].value_counts(dropna=False).sort_index().items():
            profile_rows.append({'metric': f'{field}={value}', 'value': int(count)})
    for field in FIELDS:
        profile_rows.append({'metric': f'missing_{field}', 'value': int(inventory[field].eq('').sum())})

    # Select LeafScan records and summarize their counts by species.
    leafscan_raw = inventory[inventory.Content.eq('LeafScan')].copy()
    leafscan_species = leafscan_raw.groupby(['Species', 'Genus', 'Family', 'ClassId'], dropna=False, sort=True).agg(n_images=('image_path', 'size'), n_observations=('ObservationId', lambda values: values[values.ne('')].nunique())).reset_index()
    del inventory_rows, records_by_image, jpg_by_stem


    # Export the shared manifest fields while preserving raw XML values.
    save_csv(inventory, 'inventory.csv')
    save_csv(anomaly_frame, 'inventory_anomalies.csv')
    save_csv(pd.DataFrame(xml_errors, columns=['xml_path', 'error_type', 'error']), 'xml_errors.csv')
    save_csv(pd.DataFrame([
        {'xml_path_structure': path, 'count': count, 'example_xml_path': structure_examples[path]}
        for path, count in sorted(structure.items())
    ]), 'xml_structure.csv')
    save_csv(pd.DataFrame(taxonomic_issues, columns=['key_field', 'key', 'conflicting_field', 'values', 'n_images']), 'taxonomic_issues.csv')
    save_csv(observation_consistency, 'observation_consistency.csv')
    save_csv(pd.DataFrame(profile_rows), 'inventory_profile.csv')
    save_csv(leafscan_species, 'leafscan_species_counts.csv')
    (output_dir / 'xml_examples.json').write_text(json.dumps(xml_samples, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return inventory


def select_tree_leafscans(inventory, taxonomy_file, data_dir, output_dir, config, workers=8):
    """Apply frozen botanical decisions, decode candidate images and compute hashes."""
    save_csv = partial(write_csv, output_dir=output_dir, config=config)
    criteria_input, source_extracts, source_provenance = load_taxonomic_criteria(taxonomy_file, inventory.loc[inventory.Content.eq('LeafScan'), 'Species'].unique())
    taxonomy_records = criteria_input.to_dict('records')

    # Check that distinct classes do not map to the same accepted name.
    accepted_aliases = criteria_input.loc[criteria_input.decision.eq('include') & criteria_input.accepted_name.ne('')]
    ambiguous_accepted = accepted_aliases.groupby('accepted_name').Species.nunique()
    assert not ambiguous_accepted.gt(1).any(), 'Two dataset labels share an accepted name: review the classes before continuing.'
    taxonomy = pd.DataFrame(taxonomy_records).fillna('')
    assert taxonomy.Species.is_unique, 'Exactly one taxonomic decision is required per original label.'
    assert set(taxonomy.decision) <= {'include', 'exclude', 'pending'}
    assert taxonomy.loc[taxonomy.decision.eq('include'), 'source_url'].ne('').all()
    tax_decisions = taxonomy.set_index('Species').decision.to_dict()
    exclusion_rows = []
    candidate_indices = []

    # Select tree LeafScan records and record every exclusion reason.
    for r in inventory.itertuples(index=False):
        reasons = []
        if r.xml_status != 'valid':
            reasons.append('invalid_xml_or_ambiguous_association')
        if r.Content != 'LeafScan':
            reasons.append('content_not_LeafScan' if r.Content else 'missing_content')
        if not r.Species:
            reasons.append('missing_species')
        if not r.ClassId:
            reasons.append('missing_class_id')
        if r.Content == 'LeafScan':
            decision = tax_decisions.get(r.Species, 'pending')
            if decision == 'exclude':
                reasons.append('non_tree_according_to_criteria')
            elif decision == 'pending':
                reasons.append('pending_taxonomy_or_growth_form')
        if reasons:
            exclusion_rows.append({'source_record': r._asdict()['source_record'], 'image_path': r.image_path, 'Species': r.Species, 'reason': ';'.join(reasons), 'detail': 'See the original XML and tree_species_criteria.csv.'})
        else:
            candidate_indices.append(r._asdict()['source_record'])
    candidates = inventory.loc[inventory["source_record"].isin(candidate_indices)].copy()
    assert len(candidates), 'No candidates remain with a verified tree species.'

    # Check image readability and hash both the file and decoded pixels.
    def inspect_image(row):
        result = {'source_record': row._asdict()['source_record'], 'image_path': row.image_path, 'exists': False, 'readable': False, 'error': '', 'sha256': '', 'pixel_sha256': '', 'width': None, 'height': None, 'mode': '', 'format': '', 'bytes': None, 'phash_variants': ''}
        path = resolve_image_path(row.image_path, data_dir)
        if not path.is_file():
            result['error'] = 'missing_file'
            return result
        result['exists'] = True
        try:
            data = path.read_bytes()
            result['bytes'] = len(data)
            result['sha256'] = hashlib.sha256(data).hexdigest()
            with Image.open(BytesIO(data)) as im:
                result['format'] = im.format
                im.verify()
            with Image.open(BytesIO(data)) as im:
                im.load()
                result['mode'] = im.mode
                rgb = ImageOps.exif_transpose(im).convert('RGB')
                result['width'], result['height'] = rgb.size
                prefix = f'RGB:{rgb.width}x{rgb.height}:'.encode('ascii')
                result['pixel_sha256'] = hashlib.sha256(prefix + rgb.tobytes()).hexdigest()

                # Compute perceptual hashes for rotations and reflections to find possible copies.
                small = rgb.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
                a = np.asarray(small, dtype=np.float64)
                hashes = []
                for mirrored in (False, True):
                    base = np.fliplr(a) if mirrored else a
                    for k in range(4):
                        low = dctn(np.rot90(base, k), type=2, norm='ortho')[:8, :8].ravel()
                        bits = low > np.median(low[1:])
                        bits[0] = False
                        value = 0
                        for bit in bits:
                            value = value << 1 | int(bit)
                        hashes.append(f'{value:016x}')
                result['phash_variants'] = ';'.join(hashes)
                result['readable'] = True
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        return result
    rows_to_check = list(candidates.itertuples(index=False))
    image_results = []

    # Inspect candidate images concurrently.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, record in enumerate(pool.map(inspect_image, rows_to_check), 1):
            image_results.append(record)
            if n % 1000 == 0:
                print(f'  Inspected {n:,}/{len(rows_to_check):,} candidate images.', flush=True)
    image_checks = pd.DataFrame(image_results, columns=['source_record', 'image_path', 'exists', 'readable', 'error', 'sha256', 'pixel_sha256', 'width', 'height', 'mode', 'format', 'bytes', 'phash_variants'])
    selected = candidates.merge(image_checks, on=['source_record', 'image_path'], validate='one_to_one')

    # Exclude unreadable images and reconcile all original records.
    failed_images = selected.loc[~selected['readable'].astype(bool)].copy()
    for r in failed_images.itertuples(index=False):
        exclusion_rows.append({'source_record': r._asdict()['source_record'], 'image_path': r.image_path, 'Species': r.Species, 'reason': 'unreadable_image', 'detail': r.error})
    selected = selected.loc[selected['readable'].astype(bool)].copy().reset_index(drop=True)
    assert len(selected), 'None of the candidate images can be decoded.'
    exclusions = pd.DataFrame(exclusion_rows, columns=['source_record', 'image_path', 'Species', 'reason', 'detail'])
    assert not selected['image_path'].duplicated().any()
    assert set(selected["source_record"]).isdisjoint(exclusions["source_record"])
    assert set(selected["source_record"]) | set(exclusions["source_record"]) == set(inventory["source_record"])


    # Write all filtering evidence and the readable tree LeafScan inventory.
    save_csv(taxonomy, 'tree_species_criteria.csv')
    save_csv(source_extracts, 'taxonomy_source_extracts.csv')
    save_csv(source_provenance, 'taxonomy_source_provenance.csv')
    save_csv(image_checks, 'image_checks.csv')
    save_csv(selected, 'tree_leaf_scans.csv')
    save_csv(exclusions, 'exclusions.csv')

    print(f'  Retained {len(selected):,} readable tree leaf scans from {selected.Species.nunique()} species.', flush=True)
    return selected, exclusions


def summarize_dataset(inventory, selected, output_dir, config):
    """Export counts by taxonomic rank and a noninteractive distribution chart."""
    save_csv = partial(write_csv, output_dir=output_dir, config=config)
    # Summarize the selected dataset by species, genus and family.
    by_species = species_summary(selected)
    by_genus = selected.loc[selected.Genus.ne('')].groupby('Genus').agg(**{'species_count': ('Species', 'nunique'), 'images': ('image_path', 'size')}).sort_values('species_count', ascending=False).reset_index()
    by_family = selected.loc[selected.Family.ne('')].groupby('Family').agg(**{'species_count': ('Species', 'nunique'), 'genera': ('Genus', lambda s: s[s.ne('')].nunique()), 'images': ('image_path', 'size')}).sort_values('species_count', ascending=False).reset_index()
    totals = pd.DataFrame([{'scope': label, 'images': len(frame), 'observations': frame.loc[frame.ObservationId.ne(''), 'ObservationId'].nunique(), 'species_count': frame.loc[frame.Species.ne(''), 'Species'].nunique(), 'genera': frame.loc[frame.Genus.ne(''), 'Genus'].nunique(), 'families': frame.loc[frame.Family.ne(''), 'Family'].nunique(), 'images_without_observation': int(frame.ObservationId.eq('').sum())} for label, frame in [('complete_inventory', inventory), ('leafscan', inventory.loc[inventory.Content.eq('LeafScan')]), ('readable_tree_leaf_scans', selected)]])

    plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    # Plot image versus observation counts and the distribution across species.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(by_species["observations"], by_species["images"], s=28, alpha=.65, color='#2B6A8E')
    limit = max(by_species["images"].max(), by_species["observations"].max())
    axes[0].plot([1, limit], [1, limit], '--', color='gray', linewidth=1, label='One image per observation')
    axes[0].set(xscale='log', yscale='log', xlabel='Distinct observations (log scale)',
                ylabel='Images (log scale)', title='Each point represents one species')
    axes[0].legend(fontsize=8)
    axes[1].hist(by_species["observations"], bins=min(25, max(5, len(by_species) // 4)), color='#DC8B35', edgecolor='white')
    axes[1].axvline(MIN_GROUPS, color='#992E35', linestyle='--', label=f'Reference: {MIN_GROUPS} observations')
    axes[1].set(xlabel='Distinct observations per species', ylabel='Species', title='Observation availability')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'species_observation_distribution.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


    # Save the tables underlying the exploratory summaries and figure.
    save_csv(by_species, 'selected_species_counts.csv')
    save_csv(by_genus, 'selected_genus_counts.csv')
    save_csv(by_family, 'selected_family_counts.csv')
    save_csv(totals, 'dataset_totals.csv')
    return by_species


def build_split_proposal(inventory, selected, by_species, exclusions, data_dir, output_dir, config, workers=8):
    """Keep observations and corroborated duplicate groups together in reproducible splits."""
    save_csv = partial(write_csv, output_dir=output_dir, config=config)
    uf = UnionFind(len(selected))
    edge_records = []

    # Join images from the same observation and duplicates confirmed by hashes.
    for field in ['ObservationId', 'sha256', 'pixel_sha256']:
        nonempty = selected.loc[selected[field].ne('')]
        for _, group in nonempty.groupby(field, sort=True):
            indices = group.index.tolist()
            for j in indices[1:]:
                uf.union(indices[0], j)
                if field != 'ObservationId':
                    edge_records.append((indices[0], j, field, 0, 'confirmed_duplicate'))

    # Find visually similar pairs; these are still only duplicate candidates.
    tree = BKTree()
    for i, r in selected.iterrows():
        values = [int(s, 16) for s in r.phash_variants.split(';')]
        matches = {}
        for value in set(values):
            for j, distance in tree.query(value, PHASH_DISTANCE):
                matches[j] = min(distance, matches.get(j, 65))
        for j, distance in sorted(matches.items()):
            if selected.at[i, 'sha256'] == selected.at[j, 'sha256'] or selected.at[i, 'pixel_sha256'] == selected.at[j, 'pixel_sha256']:
                continue
            edge_records.append((j, i, 'phash_8_orientations', distance, 'phash_candidate'))
        tree.add(values[0], i)

    def make_visual_thumbnail(index):
        with Image.open(resolve_image_path(selected.at[index, 'image_path'], data_dir)) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            rgb.thumbnail((VISUAL_SIZE, VISUAL_SIZE), Image.Resampling.LANCZOS)
            rgb_array = np.asarray(rgb, dtype=np.float32) / 255
            borders = np.concatenate([rgb_array[0], rgb_array[-1], rgb_array[:, 0], rgb_array[:, -1]], axis=0)
            background = np.median(borders, axis=0)
            local_mask = np.max(np.abs(rgb_array - background), axis=2) > 0.1
            left, top = ((VISUAL_SIZE - rgb.width) // 2, (VISUAL_SIZE - rgb.height) // 2)
            canvas = Image.new('RGB', (VISUAL_SIZE, VISUAL_SIZE), 'white')
            canvas.paste(rgb, (left, top))
            a = np.asarray(canvas, dtype=np.float32) / 255
            mask = np.zeros((VISUAL_SIZE, VISUAL_SIZE), dtype=bool)
            mask[top:top + rgb.height, left:left + rgb.width] = local_mask
        return (index, (a, mask))
    indices_to_verify = sorted({i for a, b, method, _, _ in edge_records if method == 'phash_8_orientations' for i in (a, b)})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        thumbnails = dict(pool.map(make_visual_thumbnail, indices_to_verify))

    # Compare foreground shape and color across rotations, reflections and small shifts.
    def verify_visual_pair(a, b):
        first, first_mask = thumbnails[a]
        second, second_mask = thumbnails[b]
        best_iou, best_mae, best_orientation, best_shift = (0.0, 1.0, '', '')
        for mirror in (False, True):
            base = np.fliplr(second) if mirror else second
            mask_base = np.fliplr(second_mask) if mirror else second_mask
            for rotation in range(4):
                other, mask_other = (np.rot90(base, rotation), np.rot90(mask_base, rotation))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        shifted_mask = np.roll(mask_other, (dy, dx), axis=(0, 1)).copy()
                        if dy:
                            region = slice(0, dy) if dy > 0 else slice(dy, None)
                            shifted_mask[region, :] = False
                        if dx:
                            region = slice(0, dx) if dx > 0 else slice(dx, None)
                            shifted_mask[:, region] = False
                        union = first_mask | shifted_mask
                        union_size = int(union.sum())
                        if union_size < 48:
                            continue
                        intersection = int((first_mask & shifted_mask).sum())
                        iou = intersection / union_size
                        if iou < best_iou and iou < VISUAL_IOU_MIN:
                            continue
                        shifted = np.roll(other, (dy, dx), axis=(0, 1)).copy()
                        if dy:
                            region = slice(0, dy) if dy > 0 else slice(dy, None)
                            shifted[region, :] = 1.0
                        if dx:
                            region = slice(0, dx) if dx > 0 else slice(dx, None)
                            shifted[:, region] = 1.0
                        mae = float(np.abs(first[union] - shifted[union]).mean())
                        if iou > best_iou or (iou == best_iou and mae < best_mae):
                            best_iou, best_mae = (iou, mae)
                            best_orientation, best_shift = (f'mirror={mirror},rotation={rotation * 90}', f'dx={dx},dy={dy}')
                        if iou >= VISUAL_IOU_MIN and mae <= VISUAL_FOREGROUND_MAE_MAX:
                            return (True, iou, mae, f'mirror={mirror},rotation={rotation * 90}', f'dx={dx},dy={dy}')
        return (False, best_iou, best_mae, best_orientation, best_shift)

    # Join candidates whose foreground shape and color meet the thresholds.
    duplicate_rows = []
    for pair_number, (a, b, method, distance, status) in enumerate(edge_records, 1):
        if method == 'phash_8_orientations':
            incorporated, iou, mae, orientation, shift = verify_visual_pair(a, b)
            status = 'possible_unconfirmed_duplicate' if incorporated else 'phash_similarity_not_supported_by_shape_and_color'
            if incorporated:
                uf.union(a, b)
        else:
            incorporated, iou, mae, orientation, shift = (True, 1.0, 0.0, '', '')
        duplicate_rows.append({'record_a': int(selected.at[a, 'source_record']), 'record_b': int(selected.at[b, 'source_record']), 'image_a': selected.at[a, 'image_path'], 'image_b': selected.at[b, 'image_path'], 'species_a': selected.at[a, 'Species'], 'species_b': selected.at[b, 'Species'], 'observation_a': selected.at[a, 'ObservationId'], 'observation_b': selected.at[b, 'ObservationId'], 'method': method, 'hamming_distance': distance, 'state': status, 'joined_group': incorporated, 'silhouette_iou': iou, 'foreground_color_mae': mae, 'compared_orientation': orientation, 'translation': shift})
        if pair_number % 1000 == 0:
            print(f'  Verified {pair_number:,}/{len(edge_records):,} duplicate links.', flush=True)
    duplicates = pd.DataFrame(duplicate_rows, columns=['record_a', 'record_b', 'image_a', 'image_b', 'species_a', 'species_b', 'observation_a', 'observation_b', 'method', 'hamming_distance', 'state', 'joined_group', 'silhouette_iou', 'foreground_color_mae', 'compared_orientation', 'translation'])

    # Assign a stable identifier to each connected image group.
    components = {}
    selected['group_id'] = ''
    for i in range(len(selected)):
        components.setdefault(uf.find(i), []).append(i)
    for indices in components.values():
        group_id = 'g_' + hashlib.sha256('\n'.join(sorted(selected.loc[indices, 'image_path'])).encode('utf-8')).hexdigest()[:20]
        selected.loc[indices, 'group_id'] = group_id
    obs_species = inventory.loc[inventory.ObservationId.ne('') & inventory.Species.ne('')].groupby('ObservationId').Species.nunique()
    obs_classes = inventory.loc[inventory.ObservationId.ne('') & inventory.ClassId.ne('')].groupby('ObservationId').ClassId.nunique()
    conflicting_observations = set(obs_species[obs_species > 1].index) | set(obs_classes[obs_classes > 1].index)

    # Quarantine groups with conflicting labels or missing observation identifiers.
    group_records = []
    for group_id, group in selected.groupby('group_id', sort=True):
        reasons = []
        if group.ObservationId.eq('').any():
            reasons.append('missing_observation_id')
        if set(group.ObservationId) & conflicting_observations:
            reasons.append('observation_id_has_multiple_species_in_inventory')
        if group.Species.nunique() != 1:
            reasons.append('component_has_multiple_species')
        if group.ClassId.nunique() != 1:
            reasons.append('component_has_multiple_class_ids')
        group_records.append({'group_id': group_id, 'images': len(group), 'observations': group.loc[group.ObservationId.ne(''), 'ObservationId'].nunique(), 'species_labels': ' | '.join(sorted(group.Species.unique())), 'quarantine_reason': ';'.join(reasons)})
    group_audit = pd.DataFrame(group_records)
    selected = selected.merge(group_audit[['group_id', 'quarantine_reason']], on='group_id', validate='many_to_one')
    usable = selected.loc[selected["quarantine_reason"].eq('')].copy()
    group_counts = usable.groupby('Species').group_id.nunique()
    eligibility = by_species[['Species', 'images', 'observations']].copy()
    eligibility['usable_groups'] = eligibility.Species.map(group_counts).fillna(0).astype(int)
    eligibility['eligible_for_split'] = eligibility["usable_groups"].ge(MIN_GROUPS)
    eligibility['reason'] = np.where(eligibility["eligible_for_split"], '', f'fewer_than_{MIN_GROUPS}_usable_groups')
    eligibility['proposed_treatment'] = np.select([eligibility["eligible_for_split"], eligibility["usable_groups"].lt(3)], ['classification_with_per_species_metrics_and_support', 'insufficient_groups_for_three_splits_collect_more_observations'], default='ineligible_review_grouped_cross_validation_or_collect_data')
    eligible_species = set(eligibility.loc[eligibility["eligible_for_split"], 'Species'])
    assert len(eligible_species) >= 2, 'There are not enough species to propose a classifier.'

    # Allocate complete groups per species with a fixed seed and minimum split counts.
    def allocate_splits(frame):
        assignments = {}
        for species in sorted(eligible_species):
            groups = sorted(frame.loc[frame.Species.eq(species), 'group_id'].unique())
            n = len(groups)
            species_seed = int.from_bytes(hashlib.sha256(f'{SEED}|{species}'.encode('utf-8')).digest()[:8], 'big')
            rng = np.random.default_rng(species_seed)
            ordered = rng.permutation(groups)
            nv = max(MIN_EVAL_GROUPS, int(np.floor(n * SPLIT_RATIOS['validation'] + 0.5)))
            nt = max(MIN_EVAL_GROUPS, int(np.floor(n * SPLIT_RATIOS['test'] + 0.5)))
            ntrain = n - nv - nt
            assert ntrain >= MIN_TRAIN_GROUPS
            for name, subset in [('train', ordered[:ntrain]), ('validation', ordered[ntrain:ntrain + nv]), ('test', ordered[ntrain + nv:])]:
                assignments.update({group_id: name for group_id in subset})
        return assignments
    assignments = allocate_splits(usable)
    assert assignments == allocate_splits(usable.sample(frac=1, random_state=SEED)), 'The partition changes when input rows are reordered.'
    proposal = selected.copy()
    proposal['split'] = proposal.group_id.map(assignments).fillna('excluded')
    proposal['exclusion_reason'] = proposal["quarantine_reason"]
    scarce = proposal.exclusion_reason.eq('') & ~proposal.Species.isin(eligible_species)
    proposal.loc[scarce, 'exclusion_reason'] = f'fewer_than_{MIN_GROUPS}_usable_groups'
    proposal['seed'] = SEED
    proposal['status'] = 'grouped_species_split'
    proposal = proposal.sort_values('image_path').reset_index(drop=True)
    active = proposal.loc[proposal.split.ne('excluded')].copy()

    # Verify coverage, minimum support and separation of groups across partitions.
    checks = []

    def record_check(name, passed, detail):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})
        assert passed, f'{name}: {detail}'
    record_check('one_row_per_image', not proposal.image_path.duplicated().any(), f'{len(proposal)} rows')
    record_check('filtered_inventory_coverage', set(proposal.image_path) == set(selected.image_path), 'Every selected image is either assigned to a split or excluded.')
    record_check('justified_exclusions', proposal.split.eq('excluded').equals(proposal.exclusion_reason.ne('')), 'Every excluded image has a reason; admitted images have none.')
    record_check('identifiers_present', active.ObservationId.ne('').all(), 'No observation identifiers are fabricated.')
    for field in ['ObservationId', 'group_id', 'sha256', 'pixel_sha256']:
        overlap_count = int(active.groupby(field).split.nunique().gt(1).sum())
        record_check(f'no_overlap_{field}', overlap_count == 0, f'{overlap_count} values cross partitions.')
    path_to_split = proposal.set_index('image_path').split.to_dict()
    broken_pairs = sum((path_to_split[r.image_a] != path_to_split[r.image_b] for r in duplicates.loc[duplicates["joined_group"]].itertuples(index=False)))
    record_check('duplicates_and_candidates_together', broken_pairs == 0, f'{broken_pairs} links have different assignment states.')
    record_check('known_labels', set(active.loc[active.split.eq('train'), 'Species']) == set(active.loc[active.split.eq('validation'), 'Species']) == set(active.loc[active.split.eq('test'), 'Species']), f'{len(eligible_species)} species are present in all three partitions.')
    split_counts = proposal.groupby(['Species', 'split']).agg(**{'images': ('image_path', 'size'), 'observations': ('ObservationId', lambda x: x[x.ne('')].nunique()), 'groups': ('group_id', 'nunique')}).reset_index()
    grid = pd.MultiIndex.from_product([sorted(selected.Species.unique()), ['train', 'validation', 'test', 'excluded']], names=['Species', 'split'])
    split_counts = split_counts.set_index(['Species', 'split']).reindex(grid, fill_value=0).reset_index()
    for species in sorted(eligible_species):
        support = split_counts.loc[split_counts.Species.eq(species)].set_index('split')["groups"]
        record_check(f'minimum_support_{species}', support.train >= MIN_TRAIN_GROUPS and support.validation >= MIN_EVAL_GROUPS and (support.test >= MIN_EVAL_GROUPS), f'train={support.train}, validation={support.validation}, test={support.test}')
    record_check('counts_reconciled', int(split_counts["images"].sum()) == len(proposal), f'{split_counts["images"].sum()} images.')
    split_totals = proposal.groupby('split').agg(**{'images': ('image_path', 'size'), 'observations': ('ObservationId', lambda x: x[x.ne('')].nunique()), 'groups': ('group_id', 'nunique'), 'species_count': ('Species', 'nunique')}).reindex(['train', 'validation', 'test', 'excluded']).fillna(0).astype(int).reset_index()
    for unit in ['images', 'observations', 'groups']:
        denom = split_totals.loc[split_totals.split.ne('excluded'), unit].sum()
        split_totals[f'active_{unit}_proportion'] = np.where(split_totals.split.eq('excluded'), np.nan, split_totals[unit] / denom)
    sensitivity = pd.DataFrame([{'minimum_groups': k, 'admitted_species': int(eligibility["usable_groups"].ge(k).sum()), 'excluded_species': int(eligibility["usable_groups"].lt(k).sum())} for k in [10, 20, 30]])

    weak = split_counts.loc[split_counts.Species.isin(eligible_species) & split_counts.split.eq('test') & split_counts["groups"].lt(10)]
    conflicts = group_audit.loc[group_audit["quarantine_reason"].ne('')]

    # Include both filtering exclusions and records excluded from the split proposal.
    proposal_exclusions = proposal.loc[proposal['split'].eq('excluded'),
        ['source_record', 'image_path', 'Species', 'exclusion_reason']].rename(
        columns={'exclusion_reason': 'reason'})
    proposal_exclusions['detail'] = 'See group_audit.csv and species_eligibility.csv.'
    all_exclusions = pd.concat([exclusions, proposal_exclusions], ignore_index=True)
    all_exclusions = all_exclusions.sort_values('source_record').reset_index(drop=True)
    assert all_exclusions['source_record'].is_unique
    assert set(active['source_record']).isdisjoint(all_exclusions['source_record'])
    assert set(active['source_record']) | set(all_exclusions['source_record']) == set(inventory['source_record'])

    save_csv(selected, 'tree_leaf_scans.csv')
    save_csv(proposal, 'split_manifest.csv')
    save_csv(all_exclusions, 'exclusions.csv')
    save_csv(duplicates, 'duplicate_audit.csv')
    save_csv(group_audit, 'group_audit.csv')
    save_csv(eligibility, 'species_eligibility.csv')
    save_csv(split_counts, 'split_counts.csv')
    save_csv(split_totals, 'split_totals.csv')
    save_csv(sensitivity, 'group_threshold_sensitivity.csv')
    save_csv(pd.DataFrame(checks), 'split_checks.csv')
    print(split_totals[['split', 'images', 'observations', 'groups', 'species_count']].to_string(index=False), flush=True)
    return proposal


def run_analysis(data_dir=ROOT / DATASET_PREFIX, output_dir=ROOT / "results" / "data" / "plantclef2015" / "v1",
                 taxonomy_file=ROOT / DATASET_PREFIX / "taxonomy_snapshot.json", workers=8,
                 *, manifest_name="split_manifest.csv"):
    """Run the complete analysis and return the final split manifest DataFrame."""
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    taxonomy_file = Path(taxonomy_file).expanduser().resolve()
    if not manifest_name or Path(manifest_name).name != manifest_name or manifest_name in {".", ".."}:
        raise ValueError("The split manifest must have a filename within the output directory.")
    if workers < 1:
        raise ValueError("--workers must be at least 1.")
    validate_paths(data_dir, output_dir, taxonomy_file)
    # Validate dataset contents before creating a directory of empty outputs.
    if not any(path.is_file() and path.suffix.lower() == ".xml" for path in data_dir.rglob("*")):
        raise ValueError(f"No XML files found in {data_dir}. Extract the original dataset, including its metadata.")
    random.seed(SEED)
    np.random.seed(SEED)
    config = build_config(workers, data_dir, taxonomy_file)
    config["manifest_filename"] = manifest_name
    # Each dataset version is written once and published only after all checks.
    # A failed preparation leaves no apparently complete output directory.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Ordinary mkdir inherits the output parent's Windows access permissions.
    # TemporaryDirectory applies private ACLs that would survive publication.
    staging_dir = output_dir.parent / f".{output_dir.name}-{uuid4().hex}"
    staging_dir.mkdir()
    try:
        (staging_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("1/4 Inventorying XML metadata and image files...", flush=True)
        inventory = inventory_dataset(data_dir, staging_dir, config)
        print("2/4 Selecting tree leaf scans and inspecting images...", flush=True)
        selected, exclusions = select_tree_leafscans(inventory, taxonomy_file, data_dir, staging_dir, config, workers)
        print("3/4 Summarizing dataset distributions...", flush=True)
        by_species = summarize_dataset(inventory, selected, staging_dir, config)
        print("4/4 Detecting duplicates and allocating observation groups...", flush=True)
        proposal = build_split_proposal(inventory, selected, by_species, exclusions, data_dir, staging_dir, config, workers)
        proposal = english_artifact_frame(proposal)
        active = proposal.loc[proposal["split"].ne("excluded")]
        taxonomy = build_taxonomy(active)
        (staging_dir / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        schema = {"schema_version": 1, "dataset": "PlantCLEF2015", "protocol": config["protocol"],
                  "columns": list(MANIFEST_COLUMNS), "path_format": "historical_dataset_prefix",
                  "taxonomy_source": "dataset_labels", "taxonomy_sha256": taxonomy_fingerprint(taxonomy),
                  "active_split_records_sha256": split_fingerprint(active),
                  "legacy_active_split_records_sha256": legacy_split_fingerprint(active)}
        (staging_dir / "manifest_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if manifest_name != "split_manifest.csv":
            target_manifest = staging_dir / manifest_name
            if target_manifest.exists():
                raise ValueError(f"The configured split filename conflicts with another preparation artifact: {manifest_name}")
            (staging_dir / "split_manifest.csv").rename(target_manifest)
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}. Choose a new --output-dir.")
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            resolved_staging = staging_dir.resolve()
            if resolved_staging.parent != output_dir.parent or staging_dir.is_symlink():
                raise ValueError("Refusing to clean a staging directory outside the output parent.")
            shutil.rmtree(resolved_staging)
    print(f"Analysis outputs saved to: {output_dir}", flush=True)
    return proposal


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="Read [data] from the experiment TOML: dataset location and output split_file.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Extracted dataset directory; overrides [data].data_dir and any environment setting "
                             "(default: PlantCLEF2015TrainingData in the project root).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override the directory of [data].split_file, keeping its filename "
                             "(default without config: results/data/plantclef2015/v1).")
    parser.add_argument("--taxonomy-file", type=Path, default=ROOT / DATASET_PREFIX / "taxonomy_snapshot.json",
                        help="Frozen botanical decisions and source extracts.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent image readers (default: 8).")
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        data = load_data_config(arguments.config)
        data_dir = arguments.data_dir if arguments.data_dir is not None else Path(data["data_dir"])
        split_file = Path(data["split_file"])
        output_dir = arguments.output_dir if arguments.output_dir is not None else split_file.parent
        run_analysis(data_dir.expanduser().resolve(), output_dir,
                     arguments.taxonomy_file, arguments.workers, manifest_name=split_file.name)
    except (OSError, ValueError, AssertionError) as error:
        parser.exit(2, f"Analysis failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

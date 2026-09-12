"""Inventory, audit, filter and split PlantCLEF2015 tree leaf scans.

The split reproduces the original experiment: tree decisions are frozen in
taxonomy_snapshot.json, and image paths use a canonical dataset prefix so that
group identifiers do not depend on the location of the downloaded data.
Legacy CSV field names and enum values are retained for experiment compatibility.
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
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent
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
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}. Use --data-dir to point to PlantCLEF2015TrainingData.")
    if not taxonomy_file.is_file():
        raise FileNotFoundError(f"Botanical source snapshot does not exist: {taxonomy_file}")
    if output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError("The output directory must not overlap the original dataset directory.")
    if output_dir == ROOT:
        raise ValueError("Choose a separate output directory instead of the project root.")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"The output path is not a directory: {output_dir}")


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
        "status": "grouped_baseline_split",
        "versions": versions,
        "data_dir": str(data_dir),
        "canonical_dataset_prefix": DATASET_PREFIX,
        "taxonomy_file": str(taxonomy_file),
        "taxonomy_snapshot_sha256": hashlib.sha256(taxonomy_file.read_bytes()).hexdigest(),
    }


def write_csv(frame, name, *, output_dir, config):
    """Write a legacy-compatible CSV without an extra DataFrame index."""
    frame.to_csv(output_dir / name, index=False, encoding=config["csv_encoding"],
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
    sources = pd.DataFrame(snapshot['extractos'])
    provenance = pd.DataFrame(snapshot['procedencia'])
    assert criteria['Species'].is_unique, 'A species label has multiple botanical decisions.'
    assert set(criteria['Species']) == set(leafscan_species_labels), 'Review the botanical table: the set of LeafScan species has changed.'
    assert set(criteria['decision']) <= {'incluir', 'excluir', 'pendiente'}
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
        if row.decision in {'incluir', 'excluir'}:
            assert normalize_taxon_label(row.Species) == normalize_taxon_label(source.full_name), f'Scientific name or author does not match: {row.Species}'
        if row.decision == 'incluir':
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
    base = frame.groupby('Species').agg(**{'imagenes': ('image_path', 'size'), 'generos': ('Genus', 'first'), 'familias': ('Family', 'first')})
    obs = frame.loc[frame.ObservationId.ne('')].groupby('Species').ObservationId.nunique()
    base['observaciones'] = obs.reindex(base.index, fill_value=0).astype(int)
    base['imagenes_sin_observacion'] = frame.ObservationId.eq('').groupby(frame.Species).sum().astype(int)
    base['imagenes_por_observacion'] = base["imagenes"] / base["observaciones"].replace(0, np.nan)
    return base.reset_index().sort_values(['observaciones', 'imagenes', 'Species'], ascending=[False, False, True])


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
        anomalies.append({'image_path': image_path, 'xml_path': xml_path, 'tipo': kind, 'campo': field, 'valor': value, 'detalle': detail})

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
        parse_status = 'valido'
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
                parse_status = 'estructura_no_esperada'
                local_notes.append(('raiz_xml_no_esperada', 'root', xml_root.tag, 'Review required; this record will not be selected automatically.'))
            for field in FIELDS:
                nodes = children.get(field, [])
                if len(nodes) > 1:
                    parse_status = 'estructura_no_esperada'
                    local_notes.append(('etiqueta_repetida', field, str(len(nodes)), 'The canonical value is left empty because it is ambiguous.'))
                    raw_values[field + '_raw'] = json.dumps([n.text or '' for n in nodes], ensure_ascii=False)
                elif nodes:
                    if len(nodes[0]):
                        parse_status = 'estructura_no_esperada'
                        local_notes.append(('metadato_anidado', field, ET.tostring(nodes[0], encoding='unicode'), 'Review required.'))
                    else:
                        raw = nodes[0].text or ''
                        raw_values[field + '_raw'] = raw
                        values[field] = raw.strip()
            for unknown in set(children) - set(FIELDS):
                local_notes.append(('etiqueta_adicional', unknown, '', 'Listed in estructura_xml.csv; do not infer its meaning.'))
            if values['MediaId'] and values['MediaId'] != xml_path.stem:
                local_notes.append(('mediaid_nombre_discrepante', 'MediaId', values['MediaId'], f'XML filename: {xml_path.stem}; association retained using matching sibling filenames.'))
        except (ET.ParseError, OSError, UnicodeError) as exc:
            parse_status = 'error_lectura'
            xml_errors.append({'xml_path': xml_rel, 'error_tipo': type(exc).__name__, 'error': str(exc)})
        for image_path in paired_paths:
            image_rel = project_relative(image_path)
            row = {'image_path': image_rel, 'xml_path': xml_rel, 'xml_status': parse_status, **values, **raw_values}
            records_by_image[image_rel].append(row)
            for kind, field, value, detail in local_notes:
                note(image_rel, xml_rel, kind, field, value, detail)
            if len(candidates) > 1:
                note(image_rel, xml_rel, 'varios_jpg_por_xml', detail=json.dumps([project_relative(p) for p in candidates]))

    # Create one row per image path, including unpaired files.
    actual_jpg = {project_relative(p) for p in jpg_files}
    inventory_rows = []
    for record_number, image_rel in enumerate(sorted(actual_jpg | set(records_by_image)), start=1):
        metadata_rows = records_by_image.get(image_rel, [])
        exists = image_rel in actual_jpg
        if not metadata_rows:
            row = {'image_path': image_rel, 'xml_path': '', 'xml_status': 'sin_xml', **{f: '' for f in FIELDS}, **{f + '_raw': '' for f in FIELDS}}
        elif len(metadata_rows) == 1:
            row = metadata_rows[0].copy()
        else:
            row = metadata_rows[0].copy()
            row['xml_path'] = json.dumps([r['xml_path'] for r in metadata_rows], ensure_ascii=False)
            row['xml_status'] = 'multiples_xml'
            for field in FIELDS:
                options = {r[field] for r in metadata_rows}
                row[field] = next(iter(options)) if len(options) == 1 else ''
                row[field + '_raw'] = json.dumps([r[field + '_raw'] for r in metadata_rows], ensure_ascii=False)
            note(image_rel, row['xml_path'], 'multiples_xml_para_imagen', detail='This conflict is not resolved automatically.')
        row['image_exists'] = exists
        row['xml_count'] = len(metadata_rows)
        row['registro_origen'] = record_number
        row['tipo_registro'] = 'imagen_y_xml' if exists and metadata_rows else 'jpg_sin_xml' if exists else 'xml_sin_jpg'
        if row['tipo_registro'] != 'imagen_y_xml':
            note(image_rel, row['xml_path'], row['tipo_registro'])
        for field in FIELDS:
            if not row[field]:
                note(image_rel, row['xml_path'], 'metadato_ausente', field, detail='Core inventory field' if field in CORE_FIELDS else 'Optional field; left empty')
        inventory_rows.append(row)
    inventory = pd.DataFrame(inventory_rows)
    first_columns = ['registro_origen', 'tipo_registro', 'image_path', 'xml_path', *FIELDS, 'image_exists', 'xml_status', 'xml_count']
    inventory = inventory[first_columns + [f + '_raw' for f in FIELDS]]
    assert inventory['image_path'].is_unique, 'The inventory must contain one row per image path.'
    assert inventory['registro_origen'].is_unique
    assert set(inventory.loc[inventory.image_exists, 'image_path']) == actual_jpg
    anomaly_frame = pd.DataFrame(anomalies, columns=['image_path', 'xml_path', 'tipo', 'campo', 'valor', 'detalle'])
    anomaly_frame.insert(0, 'registro_origen', anomaly_frame.image_path.map(inventory.set_index('image_path')["registro_origen"]))

    # Detect contradictory relationships between labels and observations.
    taxonomic_issues = []
    for source_field, target_field in [('Species', 'Genus'), ('Species', 'Family'), ('Species', 'ClassId'), ('ClassId', 'Species'), ('Genus', 'Family'), ('ObservationId', 'Species'), ('ObservationId', 'ClassId')]:
        usable = inventory[(inventory[source_field] != '') & (inventory[target_field] != '')]
        for key, group in usable.groupby(source_field, sort=True):
            alternatives = sorted(group[target_field].unique())
            if len(alternatives) > 1:
                taxonomic_issues.append({'campo_clave': source_field, 'clave': key, 'campo_conflictivo': target_field, 'valores': json.dumps(alternatives, ensure_ascii=False), 'n_imagenes': len(group)})
    genus_mismatch = inventory[(inventory.Species != '') & (inventory.Genus != '') & (inventory.Species.str.split().str[0] != inventory.Genus)]
    for species, group in genus_mismatch.groupby('Species', sort=True):
        taxonomic_issues.append({'campo_clave': 'Species', 'clave': species, 'campo_conflictivo': 'Genus_vs_primer_termino_especie', 'valores': json.dumps(sorted(group.Genus.unique()), ensure_ascii=False), 'n_imagenes': len(group)})
    observation_consistency = inventory[inventory.ObservationId.ne('')].groupby('ObservationId', sort=True).agg(**{'n_imagenes': ('image_path', 'size'), 'n_especies': ('Species', 'nunique'), 'n_clases': ('ClassId', 'nunique'), 'n_autores': ('Author', 'nunique'), 'n_fechas': ('Date', 'nunique'), 'n_ediciones': ('YearInCLEF', 'nunique'), 'n_identificadores_2014': ('ObservationId2014', lambda x: x[x.ne('')].nunique()), 'n_identificadores_2014_ausentes': ('ObservationId2014', lambda x: int(x.eq('').sum()))}).reset_index()
    profile_rows = [{'metrica': 'jpg_reales', 'valor': len(jpg_files)}, {'metrica': 'xml_reales', 'valor': len(xml_files)}, {'metrica': 'filas_inventario', 'valor': len(inventory)}, {'metrica': 'errores_xml', 'valor': len(xml_errors)}, {'metrica': 'inconsistencias_taxonomicas', 'valor': len(taxonomic_issues)}]
    for field in ['tipo_registro', 'xml_status', 'Content', 'YearInCLEF', 'LearnTag']:
        for value, count in inventory[field].value_counts(dropna=False).sort_index().items():
            profile_rows.append({'metrica': f'{field}={value}', 'valor': int(count)})
    for field in FIELDS:
        profile_rows.append({'metrica': f'ausentes_{field}', 'valor': int(inventory[field].eq('').sum())})

    # Select LeafScan records and summarize their counts by species.
    leafscan_raw = inventory[inventory.Content.eq('LeafScan')].copy()
    leafscan_species = leafscan_raw.groupby(['Species', 'Genus', 'Family', 'ClassId'], dropna=False, sort=True).agg(n_images=('image_path', 'size'), n_observations=('ObservationId', lambda values: values[values.ne('')].nunique())).reset_index()
    del inventory_rows, records_by_image, jpg_by_stem


    # Preserve the legacy inventory schema consumed by the analysis and baseline scripts.
    save_csv(inventory, 'inventario_completo.csv')
    save_csv(anomaly_frame, 'inventory_anomalies.csv')
    save_csv(pd.DataFrame(xml_errors, columns=['xml_path', 'error_tipo', 'error']), 'xml_errors.csv')
    save_csv(pd.DataFrame([
        {'xml_path_structure': path, 'count': count, 'example_xml_path': structure_examples[path]}
        for path, count in sorted(structure.items())
    ]), 'estructura_xml.csv')
    save_csv(pd.DataFrame(taxonomic_issues, columns=['campo_clave', 'clave', 'campo_conflictivo', 'valores', 'n_imagenes']), 'taxonomic_issues.csv')
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
    accepted_aliases = criteria_input.loc[criteria_input.decision.eq('incluir') & criteria_input.accepted_name.ne('')]
    ambiguous_accepted = accepted_aliases.groupby('accepted_name').Species.nunique()
    assert not ambiguous_accepted.gt(1).any(), 'Two dataset labels share an accepted name: review the classes before continuing.'
    taxonomy = pd.DataFrame(taxonomy_records).fillna('')
    assert taxonomy.Species.is_unique, 'Exactly one taxonomic decision is required per original label.'
    assert set(taxonomy.decision) <= {'incluir', 'excluir', 'pendiente'}
    assert taxonomy.loc[taxonomy.decision.eq('incluir'), 'source_url'].ne('').all()
    leaf_species_now = set(inventory.loc[inventory.Content.eq('LeafScan') & inventory.Species.ne(''), 'Species'])
    missing_decisions = sorted(leaf_species_now - set(taxonomy.Species))
    if missing_decisions:
        pass
    tax_decisions = taxonomy.set_index('Species').decision.to_dict()
    exclusion_rows = []
    candidate_indices = []

    # Select tree LeafScan records and record every exclusion reason.
    for r in inventory.itertuples(index=False):
        reasons = []
        if r.xml_status != 'valido':
            reasons.append('xml_no_valido_o_asociacion_ambigua')
        if r.Content != 'LeafScan':
            reasons.append('tipo_distinto_de_LeafScan' if r.Content else 'tipo_ausente')
        if not r.Species:
            reasons.append('especie_ausente')
        if not r.ClassId:
            reasons.append('ClassId_ausente')
        if r.Content == 'LeafScan':
            decision = tax_decisions.get(r.Species, 'pendiente')
            if decision == 'excluir':
                reasons.append('porte_no_arboreo_segun_criterio')
            elif decision == 'pendiente':
                reasons.append('taxonomia_o_porte_pendiente')
        if reasons:
            exclusion_rows.append({'registro_origen': r._asdict()['registro_origen'], 'image_path': r.image_path, 'Species': r.Species, 'motivo': ';'.join(reasons), 'detalle': 'See the original XML and criterio_especies_arboreas.csv.'})
        else:
            candidate_indices.append(r._asdict()['registro_origen'])
    candidates = inventory.loc[inventory["registro_origen"].isin(candidate_indices)].copy()
    assert len(candidates), 'No candidates remain with a verified tree species.'

    # Check image readability and hash both the file and decoded pixels.
    def inspect_image(row):
        result = {'registro_origen': row._asdict()['registro_origen'], 'image_path': row.image_path, 'exists': False, 'readable': False, 'error': '', 'sha256': '', 'pixel_sha256': '', 'width': None, 'height': None, 'mode': '', 'format': '', 'bytes': None, 'phash_variants': ''}
        path = resolve_image_path(row.image_path, data_dir)
        if not path.is_file():
            result['error'] = 'archivo_ausente'
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
    image_checks = pd.DataFrame(image_results, columns=['registro_origen', 'image_path', 'exists', 'readable', 'error', 'sha256', 'pixel_sha256', 'width', 'height', 'mode', 'format', 'bytes', 'phash_variants'])
    selected = candidates.merge(image_checks, on=['registro_origen', 'image_path'], validate='one_to_one')

    # Exclude unreadable images and reconcile all original records.
    failed_images = selected.loc[~selected['readable'].astype(bool)].copy()
    for r in failed_images.itertuples(index=False):
        exclusion_rows.append({'registro_origen': r._asdict()['registro_origen'], 'image_path': r.image_path, 'Species': r.Species, 'motivo': 'imagen_no_legible', 'detalle': r.error})
    selected = selected.loc[selected['readable'].astype(bool)].copy().reset_index(drop=True)
    assert len(selected), 'None of the candidate images can be decoded.'
    exclusions = pd.DataFrame(exclusion_rows, columns=['registro_origen', 'image_path', 'Species', 'motivo', 'detalle'])
    assert not selected['image_path'].duplicated().any()
    assert set(selected["registro_origen"]).isdisjoint(exclusions["registro_origen"])
    assert set(selected["registro_origen"]) | set(exclusions["registro_origen"]) == set(inventory["registro_origen"])


    # Write all filtering evidence and the readable tree LeafScan inventory.
    save_csv(taxonomy, 'criterio_especies_arboreas.csv')
    save_csv(source_extracts, 'taxonomy_source_extracts.csv')
    save_csv(source_provenance, 'taxonomy_source_provenance.csv')
    save_csv(image_checks, 'image_checks.csv')
    save_csv(selected, 'inventario_leafscan_arboles.csv')
    save_csv(exclusions, 'exclusiones.csv')

    print(f'  Retained {len(selected):,} readable tree leaf scans from {selected.Species.nunique()} species.', flush=True)
    return selected, exclusions


def summarize_dataset(inventory, selected, output_dir, config):
    """Export counts by taxonomic rank and a noninteractive distribution chart."""
    save_csv = partial(write_csv, output_dir=output_dir, config=config)
    # Summarize the selected dataset by species, genus and family.
    by_species = species_summary(selected)
    by_genus = selected.loc[selected.Genus.ne('')].groupby('Genus').agg(**{'especies': ('Species', 'nunique'), 'imagenes': ('image_path', 'size')}).sort_values('especies', ascending=False).reset_index()
    by_family = selected.loc[selected.Family.ne('')].groupby('Family').agg(**{'especies': ('Species', 'nunique'), 'generos': ('Genus', lambda s: s[s.ne('')].nunique()), 'imagenes': ('image_path', 'size')}).sort_values('especies', ascending=False).reset_index()
    totals = pd.DataFrame([{'ambito': label, 'imagenes': len(frame), 'observaciones': frame.loc[frame.ObservationId.ne(''), 'ObservationId'].nunique(), 'especies': frame.loc[frame.Species.ne(''), 'Species'].nunique(), 'generos': frame.loc[frame.Genus.ne(''), 'Genus'].nunique(), 'familias': frame.loc[frame.Family.ne(''), 'Family'].nunique(), 'imagenes_sin_observacion': int(frame.ObservationId.eq('').sum())} for label, frame in [('inventario_total', inventory), ('leafscan', inventory.loc[inventory.Content.eq('LeafScan')]), ('leafscan_arboles_legibles', selected)]])

    plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    # Plot image versus observation counts and the distribution across species.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(by_species["observaciones"], by_species["imagenes"], s=28, alpha=.65, color='#2B6A8E')
    limit = max(by_species["imagenes"].max(), by_species["observaciones"].max())
    axes[0].plot([1, limit], [1, limit], '--', color='gray', linewidth=1, label='One image per observation')
    axes[0].set(xscale='log', yscale='log', xlabel='Distinct observations (log scale)',
                ylabel='Images (log scale)', title='Each point represents one species')
    axes[0].legend(fontsize=8)
    axes[1].hist(by_species["observaciones"], bins=min(25, max(5, len(by_species) // 4)), color='#DC8B35', edgecolor='white')
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
                    edge_records.append((indices[0], j, field, 0, 'duplicado_confirmado'))

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
            edge_records.append((j, i, 'phash_8_orientaciones', distance, 'candidato_phash'))
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
    indices_to_verify = sorted({i for a, b, method, _, _ in edge_records if method == 'phash_8_orientaciones' for i in (a, b)})
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
        if method == 'phash_8_orientaciones':
            incorporated, iou, mae, orientation, shift = verify_visual_pair(a, b)
            status = 'posible_duplicado_no_confirmado' if incorporated else 'parecido_phash_no_corrobora_forma_color'
            if incorporated:
                uf.union(a, b)
        else:
            incorporated, iou, mae, orientation, shift = (True, 1.0, 0.0, '', '')
        duplicate_rows.append({'registro_a': int(selected.at[a, 'registro_origen']), 'registro_b': int(selected.at[b, 'registro_origen']), 'image_a': selected.at[a, 'image_path'], 'image_b': selected.at[b, 'image_path'], 'species_a': selected.at[a, 'Species'], 'species_b': selected.at[b, 'Species'], 'observation_a': selected.at[a, 'ObservationId'], 'observation_b': selected.at[b, 'ObservationId'], 'metodo': method, 'distancia_hamming': distance, 'estado': status, 'incorporado_grupo': incorporated, 'silueta_iou': iou, 'color_mae_primer_plano': mae, 'orientacion_comparada': orientation, 'desplazamiento': shift})
        if pair_number % 1000 == 0:
            print(f'  Verified {pair_number:,}/{len(edge_records):,} duplicate links.', flush=True)
    duplicates = pd.DataFrame(duplicate_rows, columns=['registro_a', 'registro_b', 'image_a', 'image_b', 'species_a', 'species_b', 'observation_a', 'observation_b', 'metodo', 'distancia_hamming', 'estado', 'incorporado_grupo', 'silueta_iou', 'color_mae_primer_plano', 'orientacion_comparada', 'desplazamiento'])

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
            reasons.append('ObservationId_ausente')
        if set(group.ObservationId) & conflicting_observations:
            reasons.append('ObservationId_con_varias_especies_en_inventario')
        if group.Species.nunique() != 1:
            reasons.append('componente_con_varias_especies')
        if group.ClassId.nunique() != 1:
            reasons.append('componente_con_varios_ClassId')
        group_records.append({'group_id': group_id, 'imagenes': len(group), 'observaciones': group.loc[group.ObservationId.ne(''), 'ObservationId'].nunique(), 'especies': ' | '.join(sorted(group.Species.unique())), 'motivo_cuarentena': ';'.join(reasons)})
    group_audit = pd.DataFrame(group_records)
    selected = selected.merge(group_audit[['group_id', 'motivo_cuarentena']], on='group_id', validate='many_to_one')
    usable = selected.loc[selected["motivo_cuarentena"].eq('')].copy()
    group_counts = usable.groupby('Species').group_id.nunique()
    eligibility = by_species[['Species', 'imagenes', 'observaciones']].copy()
    eligibility['grupos_utilizables'] = eligibility.Species.map(group_counts).fillna(0).astype(int)
    eligibility['incluida_baseline'] = eligibility["grupos_utilizables"].ge(MIN_GROUPS)
    eligibility['motivo'] = np.where(eligibility["incluida_baseline"], '', f'menos_de_{MIN_GROUPS}_grupos_utilizables')
    eligibility['tratamiento_propuesto'] = np.select([eligibility["incluida_baseline"], eligibility["grupos_utilizables"].lt(3)], ['baseline_con_metricas_por_especie_y_soporte', 'no_admite_tres_particiones_recoger_mas_observaciones'], default='fuera_baseline_revisar_validacion_cruzada_agrupada_o_recoger_datos')
    eligible_species = set(eligibility.loc[eligibility["incluida_baseline"], 'Species'])
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
    proposal['exclusion_reason'] = proposal["motivo_cuarentena"]
    scarce = proposal.exclusion_reason.eq('') & ~proposal.Species.isin(eligible_species)
    proposal.loc[scarce, 'exclusion_reason'] = f'menos_de_{MIN_GROUPS}_grupos_utilizables'
    proposal['seed'] = SEED
    proposal['status'] = 'grouped_baseline_split'
    proposal = proposal.sort_values('image_path').reset_index(drop=True)
    active = proposal.loc[proposal.split.ne('excluded')].copy()

    # Verify coverage, minimum support and separation of groups across partitions.
    checks = []

    def record_check(name, passed, detail):
        checks.append({'comprobacion': name, 'superada': bool(passed), 'detalle': detail})
        assert passed, f'{name}: {detail}'
    record_check('una_fila_por_imagen', not proposal.image_path.duplicated().any(), f'{len(proposal)} rows')
    record_check('cobertura_inventario_filtrado', set(proposal.image_path) == set(selected.image_path), 'Every selected image is either assigned to a split or excluded.')
    record_check('exclusiones_justificadas', proposal.split.eq('excluded').equals(proposal.exclusion_reason.ne('')), 'Every excluded image has a reason; admitted images have none.')
    record_check('identificadores_presentes', active.ObservationId.ne('').all(), 'No observation identifiers are fabricated.')
    for field in ['ObservationId', 'group_id', 'sha256', 'pixel_sha256']:
        overlap_count = int(active.groupby(field).split.nunique().gt(1).sum())
        record_check(f'sin_solapamiento_{field}', overlap_count == 0, f'{overlap_count} values cross partitions.')
    path_to_split = proposal.set_index('image_path').split.to_dict()
    broken_pairs = sum((path_to_split[r.image_a] != path_to_split[r.image_b] for r in duplicates.loc[duplicates["incorporado_grupo"]].itertuples(index=False)))
    record_check('duplicados_y_candidatos_juntos', broken_pairs == 0, f'{broken_pairs} links have different assignment states.')
    record_check('etiquetas_conocidas', set(active.loc[active.split.eq('train'), 'Species']) == set(active.loc[active.split.eq('validation'), 'Species']) == set(active.loc[active.split.eq('test'), 'Species']), f'{len(eligible_species)} species are present in all three partitions.')
    split_counts = proposal.groupby(['Species', 'split']).agg(**{'imagenes': ('image_path', 'size'), 'observaciones': ('ObservationId', lambda x: x[x.ne('')].nunique()), 'grupos': ('group_id', 'nunique')}).reset_index()
    grid = pd.MultiIndex.from_product([sorted(selected.Species.unique()), ['train', 'validation', 'test', 'excluded']], names=['Species', 'split'])
    split_counts = split_counts.set_index(['Species', 'split']).reindex(grid, fill_value=0).reset_index()
    for species in sorted(eligible_species):
        support = split_counts.loc[split_counts.Species.eq(species)].set_index('split')["grupos"]
        record_check(f'minimos_{species}', support.train >= MIN_TRAIN_GROUPS and support.validation >= MIN_EVAL_GROUPS and (support.test >= MIN_EVAL_GROUPS), f'train={support.train}, validation={support.validation}, test={support.test}')
    record_check('reconciliacion_recuentos', int(split_counts["imagenes"].sum()) == len(proposal), f'{split_counts["imagenes"].sum()} images.')
    split_totals = proposal.groupby('split').agg(**{'imagenes': ('image_path', 'size'), 'observaciones': ('ObservationId', lambda x: x[x.ne('')].nunique()), 'grupos': ('group_id', 'nunique'), 'especies': ('Species', 'nunique')}).reindex(['train', 'validation', 'test', 'excluded']).fillna(0).astype(int).reset_index()
    for unit in ['imagenes', 'observaciones', 'grupos']:
        denom = split_totals.loc[split_totals.split.ne('excluded'), unit].sum()
        split_totals[f'proporcion_{unit}_baseline'] = np.where(split_totals.split.eq('excluded'), np.nan, split_totals[unit] / denom)
    sensitivity = pd.DataFrame([{'minimo_grupos': k, 'especies_admitidas': int(eligibility["grupos_utilizables"].ge(k).sum()), 'especies_excluidas': int(eligibility["grupos_utilizables"].lt(k).sum())} for k in [10, 20, 30]])

    weak = split_counts.loc[split_counts.Species.isin(eligible_species) & split_counts.split.eq('test') & split_counts["grupos"].lt(10)]
    conflicts = group_audit.loc[group_audit["motivo_cuarentena"].ne('')]

    # Include both filtering exclusions and records excluded from the split proposal.
    proposal_exclusions = proposal.loc[proposal['split'].eq('excluded'),
        ['registro_origen', 'image_path', 'Species', 'exclusion_reason']].rename(
        columns={'exclusion_reason': 'motivo'})
    proposal_exclusions['detalle'] = 'See group_audit.csv and species_eligibility.csv.'
    all_exclusions = pd.concat([exclusions, proposal_exclusions], ignore_index=True)
    all_exclusions = all_exclusions.sort_values('registro_origen').reset_index(drop=True)
    assert all_exclusions['registro_origen'].is_unique
    assert set(active['registro_origen']).isdisjoint(all_exclusions['registro_origen'])
    assert set(active['registro_origen']) | set(all_exclusions['registro_origen']) == set(inventory['registro_origen'])

    save_csv(selected, 'inventario_leafscan_arboles.csv')
    save_csv(proposal, 'particion_propuesta.csv')
    save_csv(all_exclusions, 'exclusiones.csv')
    save_csv(duplicates, 'duplicate_audit.csv')
    save_csv(group_audit, 'group_audit.csv')
    save_csv(eligibility, 'species_eligibility.csv')
    save_csv(split_counts, 'split_counts.csv')
    save_csv(split_totals, 'split_totals.csv')
    save_csv(sensitivity, 'group_threshold_sensitivity.csv')
    save_csv(pd.DataFrame(checks), 'split_checks.csv')
    print(split_totals[['split', 'imagenes', 'observaciones', 'grupos', 'especies']].to_string(index=False), flush=True)
    return proposal


def run_analysis(data_dir=ROOT / DATASET_PREFIX, output_dir=ROOT / "results" / "analysis",
                 taxonomy_file=ROOT / "taxonomy_snapshot.json", workers=8):
    """Run the complete analysis and return the final split manifest DataFrame."""
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    taxonomy_file = Path(taxonomy_file).expanduser().resolve()
    if workers < 1:
        raise ValueError("--workers must be at least 1.")
    validate_paths(data_dir, output_dir, taxonomy_file)
    # Validate dataset contents before creating a directory of empty outputs.
    if not any(path.is_file() and path.suffix.lower() == ".xml" for path in data_dir.rglob("*")):
        raise ValueError(f"No XML files found in {data_dir}. Extract the original dataset, including its metadata.")
    random.seed(SEED)
    np.random.seed(SEED)
    config = build_config(workers, data_dir, taxonomy_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("1/4 Inventorying XML metadata and image files...", flush=True)
    inventory = inventory_dataset(data_dir, output_dir, config)
    print("2/4 Selecting tree leaf scans and inspecting images...", flush=True)
    selected, exclusions = select_tree_leafscans(inventory, taxonomy_file, data_dir, output_dir, config, workers)
    print("3/4 Summarizing dataset distributions...", flush=True)
    by_species = summarize_dataset(inventory, selected, output_dir, config)
    print("4/4 Detecting duplicates and allocating observation groups...", flush=True)
    proposal = build_split_proposal(inventory, selected, by_species, exclusions, data_dir, output_dir, config, workers)
    print(f"Analysis outputs saved to: {output_dir}", flush=True)
    return proposal


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / DATASET_PREFIX,
                        help="Extracted PlantCLEF2015TrainingData directory (default: beside this script).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "analysis",
                        help="Analysis output directory (default: results/analysis beside this script).")
    parser.add_argument("--taxonomy-file", type=Path, default=ROOT / "taxonomy_snapshot.json",
                        help="Frozen botanical decisions and source extracts.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent image readers (default: 8).")
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        run_analysis(arguments.data_dir, arguments.output_dir, arguments.taxonomy_file, arguments.workers)
    except (OSError, ValueError, AssertionError) as error:
        parser.exit(2, f"Analysis failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

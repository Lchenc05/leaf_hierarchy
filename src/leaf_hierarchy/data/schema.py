"""English names for PlantCLEF outputs and documented historical adaptations.

Raw XML text, scientific names, source citations and frozen evidence are never
translated. Only generated field names and controlled audit codes are adapted.
"""

from __future__ import annotations

import re

import pandas as pd

from .manifest import normalize_manifest_columns

OUTPUT_FILENAMES = {
    "inventario_completo.csv": "inventory.csv",
    "inventario_leafscan_arboles.csv": "tree_leaf_scans.csv",
    "particion_propuesta.csv": "split_manifest.csv",
    "exclusiones.csv": "exclusions.csv",
    "criterio_especies_arboreas.csv": "tree_species_criteria.csv",
    "estructura_xml.csv": "xml_structure.csv",
}

# The XML tags remain untouched inside the reader, and become common headers
# only at the export boundary. Their corresponding *_raw values are unchanged.
XML_COLUMNS = {
    "ObservationId": "observation_id", "Species": "species", "Genus": "genus",
    "Family": "family", "Content": "content", "ClassId": "class_id",
    "MediaId": "media_id", "Author": "author", "Date": "date", "Location": "location",
    "Latitude": "latitude", "Longitude": "longitude", "YearInCLEF": "year_in_clef",
    "ObservationId2014": "observation_id_2014", "ImageId2014": "image_id_2014",
    "LearnTag": "learn_tag", "Vote": "vote",
}

LEGACY_NAMES = {
    "registro_origen": "source_record", "tipo_registro": "record_type",
    "registro_a": "record_a", "registro_b": "record_b",
    "tipo": "type", "campo": "field", "valor": "value", "detalle": "detail",
    "campo_clave": "key_field", "clave": "key", "campo_conflictivo": "conflicting_field",
    "valores": "values", "error_tipo": "error_type", "metrica": "metric",
    "n_imagenes": "n_images", "n_especies": "n_species", "n_clases": "n_classes",
    "n_autores": "n_authors", "n_fechas": "n_dates", "n_ediciones": "n_editions",
    "n_identificadores_2014": "n_observation_ids_2014",
    "n_identificadores_2014_ausentes": "n_missing_observation_ids_2014",
    "imagenes": "images", "observaciones": "observations", "especies": "species_count",
    "generos": "genera", "familias": "families", "grupos": "groups",
    "imagenes_sin_observacion": "images_without_observation",
    "imagenes_por_observacion": "images_per_observation", "ambito": "scope",
    "metodo": "method", "distancia_hamming": "hamming_distance", "estado": "state",
    "incorporado_grupo": "joined_group", "silueta_iou": "silhouette_iou",
    "color_mae_primer_plano": "foreground_color_mae", "orientacion_comparada": "compared_orientation",
    "desplazamiento": "translation", "motivo_cuarentena": "quarantine_reason",
    "fase": "stage", "metadatos_ausentes": "missing_metadata", "error_xml": "xml_error",
    "grupos_utilizables": "usable_groups", "incluida_baseline": "eligible_for_split",
    "motivo": "reason", "tratamiento_propuesto": "proposed_treatment",
    "comprobacion": "check", "superada": "passed", "minimo_grupos": "minimum_groups",
    "especies_admitidas": "admitted_species", "especies_excluidas": "excluded_species",
    "porte": "growth_form",
    "proporcion_imagenes_baseline": "active_image_proportion",
    "proporcion_observaciones_baseline": "active_observation_proportion",
    "proporcion_grupos_baseline": "active_group_proportion",
}

LEGACY_CODES = {
    "valido": "valid", "estructura_no_esperada": "unexpected_structure",
    "raiz_xml_no_esperada": "unexpected_xml_root", "etiqueta_repetida": "repeated_tag",
    "metadato_anidado": "nested_metadata", "etiqueta_adicional": "additional_tag",
    "mediaid_nombre_discrepante": "media_id_filename_mismatch", "error_lectura": "read_error",
    "varios_jpg_por_xml": "multiple_jpg_per_xml", "sin_xml": "missing_xml",
    "multiples_xml": "multiple_xml", "multiples_xml_para_imagen": "multiple_xml_per_image",
    "imagen_y_xml": "image_and_xml", "jpg_sin_xml": "jpg_without_xml", "xml_sin_jpg": "xml_without_jpg",
    "metadato_ausente": "missing_metadata", "Genus_vs_primer_termino_especie": "genus_vs_species_first_token",
    "jpg_reales": "actual_jpg_files", "xml_reales": "actual_xml_files",
    "filas_inventario": "inventory_rows", "errores_xml": "xml_errors",
    "inconsistencias_taxonomicas": "taxonomic_inconsistencies",
    "incluir": "include", "excluir": "exclude", "pendiente": "pending",
    "xml_no_valido_o_asociacion_ambigua": "invalid_xml_or_ambiguous_association",
    "tipo_distinto_de_LeafScan": "content_not_LeafScan", "tipo_ausente": "missing_content",
    "especie_ausente": "missing_species", "ClassId_ausente": "missing_class_id",
    "porte_no_arboreo_segun_criterio": "non_tree_according_to_criteria",
    "taxonomia_o_porte_pendiente": "pending_taxonomy_or_growth_form",
    "archivo_ausente": "missing_file", "imagen_no_legible": "unreadable_image",
    "inventario_total": "complete_inventory", "leafscan_arboles_legibles": "readable_tree_leaf_scans",
    "duplicado_confirmado": "confirmed_duplicate", "phash_8_orientaciones": "phash_8_orientations",
    "candidato_phash": "phash_candidate", "posible_duplicado_no_confirmado": "possible_unconfirmed_duplicate",
    "parecido_phash_no_corrobora_forma_color": "phash_similarity_not_supported_by_shape_and_color",
    "ObservationId_ausente": "missing_observation_id",
    "ObservationId_con_varias_especies_en_inventario": "observation_id_has_multiple_species_in_inventory",
    "componente_con_varias_especies": "component_has_multiple_species",
    "componente_con_varios_ClassId": "component_has_multiple_class_ids",
    "baseline_con_metricas_por_especie_y_soporte": "classification_with_per_species_metrics_and_support",
    "no_admite_tres_particiones_recoger_mas_observaciones": "insufficient_groups_for_three_splits_collect_more_observations",
    "fuera_baseline_revisar_validacion_cruzada_agrupada_o_recoger_datos": "ineligible_review_grouped_cross_validation_or_collect_data",
    "una_fila_por_imagen": "one_row_per_image", "cobertura_inventario_filtrado": "filtered_inventory_coverage",
    "exclusiones_justificadas": "justified_exclusions", "identificadores_presentes": "identifiers_present",
    "duplicados_y_candidatos_juntos": "duplicates_and_candidates_together",
    "etiquetas_conocidas": "known_labels", "reconciliacion_recuentos": "counts_reconciled",
    "WCVP_binomio_y_autor_normalizado": "WCVP_normalized_binomial_and_author",
    "binomio_y_autor_normalizado": "normalized_binomial_and_author", "pendiente_autoria": "pending_author_match",
    "sin verificar": "unverified", "árbol (puede tener porte variable)": "tree (growth form may vary)",
    "grouped_baseline_split": "grouped_species_split",
    "seleccion": "selection",
    "propuesta_pendiente_revision_profesor": "proposed_pending_instructor_review",
    "propuesta_para_revision_profesor": "proposed_for_instructor_review",
}


def translate_code(value):
    """Translate known audit codes, including compound exclusion/check labels."""
    if not isinstance(value, str):
        return value
    if value in LEGACY_CODES:
        return LEGACY_CODES[value]
    parts = value.split(";")
    if len(parts) > 1:
        return ";".join(translate_code(part) for part in parts)
    if re.fullmatch(r"menos_de_\d+_grupos_utilizables", value):
        return f"fewer_than_{value.split('_')[2]}_usable_groups"
    for old, new in (("sin_solapamiento_", "no_overlap_"), ("minimos_", "minimum_support_"), ("ausentes_", "missing_")):
        if value.startswith(old):
            suffix = value[len(old):]
            return new + XML_COLUMNS.get(suffix, suffix)
    if "=" in value:
        field, content = value.split("=", 1)
        return f"{XML_COLUMNS.get(field, LEGACY_NAMES.get(field, field))}={translate_code(content)}"
    for prefix in ("no_overlap_", "missing_"):
        if value.startswith(prefix):
            suffix = value[len(prefix):]
            return prefix + XML_COLUMNS.get(suffix, suffix)
    return XML_COLUMNS.get(value, LEGACY_NAMES.get(value, value))


def english_artifact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate generated schemas and codes while preserving source content."""
    result = normalize_manifest_columns(frame)
    names = {**XML_COLUMNS, **{f"{key}_raw": f"{value}_raw" for key, value in XML_COLUMNS.items()}, **LEGACY_NAMES}
    result = result.rename(columns={name: names.get(name, name) for name in result.columns})
    if not result.columns.is_unique:
        raise ValueError("The translated artifact contains conflicting column aliases.")
    for field in ("xml_status", "record_type", "type", "field", "key_field", "conflicting_field",
                  "metric", "decision", "decision_review_status", "match_method", "growth_form", "error", "scope", "method",
                  "state", "quarantine_reason", "reason", "exclusion_reason", "proposed_treatment", "check", "status",
                  "stage", "missing_metadata"):
        if field in result:
            result[field] = result[field].map(translate_code)
    # These generated links point at renamed artifacts; quoted evidence/notes do not.
    if "detail" in result:
        for old, new in OUTPUT_FILENAMES.items():
            result["detail"] = result["detail"].map(lambda value: value.replace(old, new) if isinstance(value, str) else value)
    return result

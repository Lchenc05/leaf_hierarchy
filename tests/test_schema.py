"""Historical artifact translation preserves dataset and evidence identity."""

import unittest

import pandas as pd

from leaf_hierarchy.data.manifest import legacy_split_fingerprint, split_fingerprint
from leaf_hierarchy.data.schema import english_artifact_frame


class ArtifactSchemaTests(unittest.TestCase):
    def test_historical_translation_preserves_records_and_source_text(self):
        source = pd.DataFrame([{
            "registro_origen": "original_record",
            "image_path": "PlantCLEF2015TrainingData/train/100373.jpg",
            "Species": "Prunus spinosa L.", "Genus": "Prunus", "Family": "Rosaceae",
            "ClassId": "123", "ObservationId": "456", "group_id": "stable_group",
            "split": "test", "Content": "LeafScan",
            "fase": "seleccion", "metadatos_ausentes": "Location;ObservationId2014;ImageId2014",
            "error_xml": "", "status": "propuesta_pendiente_revision_profesor",
            "decision_review_status": "propuesta_para_revision_profesor",
            "exclusion_reason": "componente_con_varias_especies;componente_con_varios_ClassId",
            "Species_raw": " Prunus spinosa L. ", "Location_raw": "seleccion",
            "evidence": "Original: seleccion; propuesta_pendiente_revision_profesor",
            "notes": "See inventario_completo.csv in the original evidence.",
            "detail": "inventario_completo.csv; particion_propuesta.csv",
        }])
        original = source.copy(deep=True)

        translated = english_artifact_frame(source)

        self.assertEqual(translated.loc[0, "stage"], "selection")
        self.assertEqual(translated.loc[0, "missing_metadata"],
                         "location;observation_id_2014;image_id_2014")
        self.assertEqual(translated.loc[0, "xml_error"], "")
        self.assertEqual(translated.loc[0, "status"], "proposed_pending_instructor_review")
        self.assertEqual(translated.loc[0, "decision_review_status"], "proposed_for_instructor_review")
        self.assertEqual(translated.loc[0, "exclusion_reason"],
                         "component_has_multiple_species;component_has_multiple_class_ids")
        self.assertEqual(translated.loc[0, "detail"], "inventory.csv; split_manifest.csv")
        for before, after in (("Species", "species"), ("Species_raw", "species_raw"),
                              ("Location_raw", "location_raw"), ("evidence", "evidence"),
                              ("notes", "notes"), ("image_path", "image_path"),
                              ("group_id", "group_id")):
            self.assertEqual(source.loc[0, before], translated.loc[0, after])
        for fingerprint in (split_fingerprint, legacy_split_fingerprint):
            self.assertEqual(fingerprint(source), fingerprint(translated))
        pd.testing.assert_frame_equal(source, original)
        pd.testing.assert_frame_equal(english_artifact_frame(translated), translated)


if __name__ == "__main__":
    unittest.main()

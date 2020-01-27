"""
Set of anatomical atlases for convenience.

(c) neurolib-devs
"""

import logging


class BaseAtlas:
    """
    Simple base for anatomical atlases. Internal representation as dictionary
    {node_num: "name"}.
    """

    name = ""
    label = ""

    def __init__(self, atlas):
        """
        :param atlas: anatomical atlas / parcellation
        :type atlas: dict
        """
        if sorted(atlas)[0] != 0:
            logging.warning("Atlas doesn't start at 0, reindexing...")
            reindexed = {}
            for new_idx, old_idx in enumerate(sorted(atlas)):
                reindexed[new_idx] = atlas[old_idx]
            atlas = reindexed
        self.atlas = atlas

    def __len__(self):
        return len(self.atlas)

    def __getitem__(self, item):
        return self.atlas[item]

    def __str__(self):
        return f"{self.name} atlas with {self.no_rois} ROIs."

    @property
    def node_names(self):
        """
        Return node names in the correct order, i.e. sorted.
        """
        return [self[key] for key in sorted(self.atlas)]

    @property
    def no_rois(self):
        """
        Return number of ROIs in the atlas.
        """
        return len(self.atlas)

    def add_rois(self, extra_rois):
        """
        Add additional ROIs to the atlas.

        :param extra_rois: ROIs to add to the atlas, must have unique keys
        :type extra_rois: dict
        """
        for key in extra_rois:
            assert key not in self.atlas, f"Node {key} already exists"
        self.atlas.update(extra_rois)

    def remove_rois(self, rois_to_remove, reindex=False):
        """
        Remove some ROIs from the atlas. Optionally re-index.

        :param rois_to_remove: list of ROIs to remove
        :type rois_to_remove: list
        :param reindex: whether to reindex ROIs that are still in the atlas
        :type reindex: bool
        """
        for key in rois_to_remove:
            res = self.atlas.pop(key, None)
            if res is None:
                logging.warning(f"Node {key} not found, doing nothing...")
        if reindex:
            reindexed = {}
            # new indices as order in sorted old dict by keys
            for new_idx, old_idx in enumerate(sorted(self.atlas)):
                reindexed[new_idx] = self.atlas[old_idx]
            self.atlas = reindexed


class AutomatedAnatomicalParcellation2(BaseAtlas):
    """
    AAL2 atlas.

    Rolls ET, Joliot M, Tzourio-Mazoyer N (2015) Implementation of a new
        parcellation of the orbitofrontal cortex in the automated anatomical
        abeling atlas. NeuroImage 10.1016/j.neuroimage.2015.07.075.
    """

    name = "Automated anatomical Parcellation 2"
    label = "AAL2"

    aal2 = {
        1: "Precentral_L",
        2: "Precentral_R",
        3: "Frontal_Sup_2_L",
        4: "Frontal_Sup_2_R",
        5: "Frontal_Mid_2_L",
        6: "Frontal_Mid_2_R",
        7: "Frontal_Inf_Oper_L",
        8: "Frontal_Inf_Oper_R",
        9: "Frontal_Inf_Tri_L",
        10: "Frontal_Inf_Tri_R",
        11: "Frontal_Inf_Orb_2_L",
        12: "Frontal_Inf_Orb_2_R",
        13: "Rolandic_Oper_L",
        14: "Rolandic_Oper_R",
        15: "Supp_Motor_Area_L",
        16: "Supp_Motor_Area_R",
        17: "Olfactory_L",
        18: "Olfactory_R",
        19: "Frontal_Sup_Medial_L",
        20: "Frontal_Sup_Medial_R",
        21: "Frontal_Med_Orb_L",
        22: "Frontal_Med_Orb_R",
        23: "Rectus_L",
        24: "Rectus_R",
        25: "OFCmed_L",
        26: "OFCmed_R",
        27: "OFCant_L",
        28: "OFCant_R",
        29: "OFCpost_L",
        30: "OFCpost_R",
        31: "OFClat_L",
        32: "OFClat_R",
        33: "Insula_L",
        34: "Insula_R",
        35: "Cingulate_Ant_L",
        36: "Cingulate_Ant_R",
        37: "Cingulate_Mid_L",
        38: "Cingulate_Mid_R",
        39: "Cingulate_Post_L",
        40: "Cingulate_Post_R",
        41: "Hippocampus_L",
        42: "Hippocampus_R",
        43: "ParaHippocampal_L",
        44: "ParaHippocampal_R",
        45: "Amygdala_L",
        46: "Amygdala_R",
        47: "Calcarine_L",
        48: "Calcarine_R",
        49: "Cuneus_L",
        50: "Cuneus_R",
        51: "Lingual_L",
        52: "Lingual_R",
        53: "Occipital_Sup_L",
        54: "Occipital_Sup_R",
        55: "Occipital_Mid_L",
        56: "Occipital_Mid_R",
        57: "Occipital_Inf_L",
        58: "Occipital_Inf_R",
        59: "Fusiform_L",
        60: "Fusiform_R",
        61: "Postcentral_L",
        62: "Postcentral_R",
        63: "Parietal_Sup_L",
        64: "Parietal_Sup_R",
        65: "Parietal_Inf_L",
        66: "Parietal_Inf_R",
        67: "SupraMarginal_L",
        68: "SupraMarginal_R",
        69: "Angular_L",
        70: "Angular_R",
        71: "Precuneus_L",
        72: "Precuneus_R",
        73: "Paracentral_Lobule_L",
        74: "Paracentral_Lobule_R",
        75: "Caudate_L",
        76: "Caudate_R",
        77: "Putamen_L",
        78: "Putamen_R",
        79: "Pallidum_L",
        80: "Pallidum_R",
        81: "Thalamus_L",
        82: "Thalamus_R",
        83: "Heschl_L",
        84: "Heschl_R",
        85: "Temporal_Sup_L",
        86: "Temporal_Sup_R",
        87: "Temporal_Pole_Sup_L",
        88: "Temporal_Pole_Sup_R",
        89: "Temporal_Mid_L",
        90: "Temporal_Mid_R",
        91: "Temporal_Pole_Mid_L",
        92: "Temporal_Pole_Mid_R",
        93: "Temporal_Inf_L",
        94: "Temporal_Inf_R",
        95: "Cerebelum_Crus1_L",
        96: "Cerebelum_Crus1_R",
        97: "Cerebelum_Crus2_L",
        98: "Cerebelum_Crus2_R",
        99: "Cerebelum_3_L",
        100: "Cerebelum_3_R",
        101: "Cerebelum_4_5_L",
        102: "Cerebelum_4_5_R",
        103: "Cerebelum_6_L",
        104: "Cerebelum_6_R",
        105: "Cerebelum_7b_L",
        106: "Cerebelum_7b_R",
        107: "Cerebelum_8_L",
        108: "Cerebelum_8_R",
        109: "Cerebelum_9_L",
        110: "Cerebelum_9_R",
        111: "Cerebelum_10_L",
        112: "Cerebelum_10_R",
        113: "Vermis_1_2",
        114: "Vermis_3",
        115: "Vermis_4_5",
        116: "Vermis_6",
        117: "Vermis_7",
        118: "Vermis_8",
        119: "Vermis_9",
        120: "Vermis_10",
    }

    def __init__(self):
        super().__init__(self.aal2)
        # define convenience regions - indexing from 0!!
        self.cerebellum = [region for region in range(94, 120)]
        self.thalamus = [80, 81]
        self.basal_ganglia = [region for region in range(74, 80)]
        self.amygdala = [44, 45]
        self.hippocampus = [region for region in range(40, 44)]
        self.subcortical = (
            self.cerebellum
            + self.thalamus
            + self.basal_ganglia
            + self.amygdala
            + self.hippocampus
        )


class DesikanKilliany(BaseAtlas):
    """
    Desikan-Killiany atlas.

    Desikan, R. S., Ségonne, F., Fischl, B., Quinn, B. T., Dickerson, B. C.,
        Blacker, D., et al. (2006). An automated labeling system for
        subdividing the human cerebral cortex on MRI scans into gyral based
        regions of interest. NeuroImage, 31(3), 968–980.
        http://doi.org/10.1016/j.neuroimage.2006.01.021
    """

    name = "Desikan-Killiany"
    label = "DK"

    dk = {
        1: "Banks_superior_temporal_sulcus_L",
        35: "Banks_superior_temporal_sulcus_R",
        2: "Caudal_anterior_cingulate_cortex_L",
        36: "Caudal_anterior_cingulate_cortex_R",
        3: "Caudal_middle_frontal_gyrus_L",
        37: "Caudal_middle_frontal_gyrus_R",
        4: "Cuneus_cortex_L",
        38: "Cuneus_cortex_R",
        5: "Entorhinal_cortex_L",
        39: "Entorhinal_cortex_R",
        6: "Fusiform_gyrus_L",
        40: "Fusiform_gyrus_R",
        7: "Inferior_parietal_cortex_L",
        41: "Inferior_parietal_cortex_R",
        8: "Inferior_temporal_gyrus_L",
        42: "Inferior_temporal_gyrus_R",
        9: "Isthmus_cingulate_cortex_L",
        43: "Isthmus_cingulate_cortex_R",
        10: "Lateral_occipital_cortex_L",
        44: "Lateral_occipital_cortex_R",
        11: "Lateral_orbital_frontal_cortex_L",
        45: "Lateral_orbital_frontal_cortex_R",
        12: "Lingual_gyrus_L",
        46: "Lingual_gyrus_R",
        13: "Medial_orbital_frontal_cortex_L",
        47: "Medial_orbital_frontal_cortex_R",
        14: "Middle_temporal_gyrus_L",
        48: "Middle_temporal_gyrus_R",
        15: "Parahippocampal_gyrus_L",
        49: "Parahippocampal_gyrus_R",
        16: "Paracentral_lobule_L",
        50: "Paracentral_lobule_R",
        17: "Pars_opercularis_L",
        51: "Pars_opercularis_R",
        18: "Pars_orbitalis_L",
        52: "Pars_orbitalis_R",
        19: "Pars_triangularis_L",
        53: "Pars_triangularis_R",
        20: "Pericalcarine_cortex_L",
        54: "Pericalcarine_cortex_R",
        21: "Postcentral_gyrus_L",
        55: "Postcentral_gyrus_R",
        22: "Posterior_cingulate_cortex_L",
        56: "Posterior_cingulate_cortex_R",
        23: "Precentral_gyrus_L",
        57: "Precentral_gyrus_R",
        24: "Precuneus_cortex_L",
        58: "Precuneus_cortex_R",
        25: "Rostral_anterior_cingulate_cortex_L",
        59: "Rostral_anterior_cingulate_cortex_R",
        26: "Rostral_middle_frontal_gyrus_L",
        60: "Rostral_middle_frontal_gyrus_R",
        27: "Superior_frontal_cortex_L",
        61: "Superior_frontal_cortex_R",
        28: "Superior_parietal_cortex_L",
        62: "Superior_parietal_cortex_R",
        29: "Superior_temporal_gyrus_L",
        63: "Superior_temporal_gyrus_R",
        30: "Supramarginal_gyrus_L",
        64: "Supramarginal_gyrus_R",
        31: "Frontal_pole_L",
        65: "Frontal_pole_R",
        32: "Temporal_pole_L",
        66: "Temporal_pole_R",
        33: "Transverse_temporal_cortex_L",
        67: "Transverse_temporal_cortex_R",
        34: "Insula_L",
        68: "Insula_R",
    }

    def __init__(self):
        super().__init__(self.dk)

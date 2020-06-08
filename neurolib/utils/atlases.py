import logging
import numpy as np


class BaseAtlas:
    """
    Simple base for anatomical atlases. Internal representation as dictionary
    {node_num: "name"}.
    """

    name = ""
    label = ""

    def __init__(self, atlas, coords=None):
        """
        :param atlas: anatomical atlas / parcellation
        :type atlas: dict
        :param coords: list of coordinates of regions
        :type coords: list[list,..]
        """
        if sorted(atlas)[0] != 0:
            logging.warning("Atlas doesn't start at 0, reindexing...")
            reindexed = {}
            for new_idx, old_idx in enumerate(sorted(atlas)):
                reindexed[new_idx] = atlas[old_idx]
            atlas = reindexed
        self.atlas = atlas

        self._coordinates = coords

    def __len__(self):
        return len(self.atlas)

    def __getitem__(self, item):
        return self.atlas[item]

    def __str__(self):
        return f"{self.name} atlas with {self.no_rois} ROIs."

    def names(self, group="cortex"):
        return [self.atlas[i] for i in getattr(self, group)]

    def coords(self, group="cortex"):
        if self._coordinates is not None:
            return [[self._coordinates[k][i] for k in range(3)] for i in getattr(self, group)]

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

    # geometric center of each region in x, y, z
    # extracted from caglorithm's brain (Creative Commons license)
    aal2_centers = np.array(
        [
            [
                71.31516853,
                173.95768398,
                95.24989269,
                151.19834526,
                74.52836592,
                171.543507,
                55.58722414,
                185.04809904,
                59.18980643,
                185.31195327,
                61.43053161,
                179.73808422,
                56.81794636,
                187.1461036,
                115.24512323,
                133.13072148,
                105.2196235,
                129.61279269,
                114.85255091,
                133.5019304,
                110.64765988,
                128.83632218,
                108.86728656,
                127.13277537,
                96.69134934,
                139.31019614,
                83.43701538,
                156.56297549,
                77.19295302,
                161.61877598,
                59.25373134,
                182.65699208,
                71.14166567,
                169.08800157,
                113.28624173,
                130.49484354,
                112.94762641,
                130.19460512,
                112.57996093,
                127.34814675,
                83.51110075,
                153.49899146,
                87.63039921,
                147.49505571,
                85.33577564,
                151.3536401,
                108.14173754,
                138.61021223,
                112.03682751,
                136.98237904,
                96.78036518,
                136.69068417,
                97.91694408,
                150.78151187,
                75.28163615,
                166.09355205,
                67.0816092,
                164.85827664,
                73.66615272,
                158.48139379,
                66.33153749,
                173.05648263,
                91.61978056,
                155.09553994,
                66.39655524,
                179.86147147,
                48.44893442,
                192.4856517,
                62.75728204,
                177.96237714,
                111.58202079,
                133.15252374,
                112.80247129,
                131.96000555,
                101.46164333,
                137.59898836,
                85.10126375,
                154.06777031,
                93.14814815,
                144.63333333,
                103.48633996,
                133.37664152,
                62.82978723,
                177.85610766,
                48.52667194,
                192.98045395,
                61.89922101,
                180.94243156,
                43.65812985,
                191.32951518,
                64.65809751,
                175.2343377,
                49.06896014,
                185.02990309,
                66.39794063,
                163.73326141,
                75.76868545,
                155.3089584,
                103.93187232,
                131.35811014,
                95.78770626,
                137.76860677,
                84.51427994,
                147.16997392,
                70.01577929,
                155.62003109,
                78.73889922,
                145.05210579,
                98.16479302,
                125.34246877,
                84.17020425,
                147.90809859,
                116.16666667,
                117.53409407,
                117.98524526,
                117.21655151,
                116.06942393,
                114.97199908,
                114.58530184,
                114.6412116,
            ],
            [
                133.91200634,
                130.63284367,
                196.31406388,
                191.86604369,
                193.63212808,
                192.56488101,
                168.03704544,
                169.63131039,
                193.00851424,
                192.21824031,
                202.61166655,
                204.95174192,
                141.20358006,
                144.82183274,
                144.92727889,
                138.74857687,
                181.88438867,
                183.25623453,
                213.45355882,
                215.19441506,
                232.23286888,
                229.15061318,
                213.6534235,
                212.17121782,
                213.87581634,
                215.55833615,
                225.82496504,
                227.64623838,
                195.97493091,
                197.40326791,
                216.09259259,
                211.56669598,
                164.02235089,
                165.87844599,
                200.74702804,
                201.64838667,
                124.10527355,
                133.5334746,
                89.90985208,
                93.5612708,
                132.40755201,
                135.35293071,
                142.2248582,
                144.40181371,
                162.00682163,
                163.97515662,
                49.62336828,
                58.59151082,
                40.78278136,
                43.74940442,
                67.42754697,
                70.66225844,
                34.88369643,
                41.2640723,
                44.48127974,
                46.52850484,
                54.41367816,
                51.17324263,
                110.5576603,
                111.8914675,
                112.06795841,
                106.75631203,
                59.80065267,
                59.25726796,
                80.89854555,
                81.63707702,
                102.83413981,
                105.91200125,
                65.04195063,
                66.99809772,
                66.82652807,
                68.76876363,
                101.79076977,
                93.95286615,
                169.20790729,
                171.83036006,
                160.75307768,
                163.93729953,
                156.40722222,
                158.29888268,
                130.1727437,
                131.88220173,
                127.38728791,
                132.08281573,
                128.40333495,
                126.5464879,
                183.83029487,
                181.41831723,
                112.74128064,
                108.49340386,
                187.67814524,
                185.33581048,
                127.70758198,
                122.49721581,
                76.66069012,
                76.20414207,
                69.93301962,
                76.24374932,
                112.08956646,
                116.29250677,
                104.42495651,
                105.87311566,
                85.25084198,
                86.79816193,
                90.90949812,
                85.99300466,
                98.02760106,
                95.10999527,
                104.19177357,
                103.01322557,
                123.96395675,
                123.34260563,
                109.78390805,
                106.23406624,
                88.7930428,
                72.10568467,
                68.07237814,
                80.26692678,
                93.28638087,
                104.73976109,
            ],
            [
                173.28640568,
                168.65816171,
                172.29010226,
                170.50384851,
                166.31060777,
                160.42796984,
                145.6567198,
                141.26061316,
                146.46240327,
                139.17263954,
                124.09846239,
                118.35301117,
                131.8744676,
                126.20448063,
                187.36563814,
                185.52072225,
                108.43221145,
                107.76413478,
                169.46412865,
                168.60826105,
                128.59842801,
                127.1576385,
                109.17926579,
                107.68409107,
                108.52215412,
                106.53593169,
                120.48125079,
                116.47455621,
                106.20568496,
                102.39282607,
                119.79325594,
                109.69451774,
                124.80022201,
                117.44317996,
                144.85418104,
                146.57548798,
                158.51751818,
                157.69512055,
                127.87510466,
                125.05257186,
                96.82570946,
                93.9790579,
                85.45206152,
                83.81491486,
                95.64729661,
                93.08295528,
                91.02415447,
                97.24077687,
                114.99554174,
                117.05130206,
                82.61423657,
                84.05270349,
                114.47056593,
                118.20361515,
                103.20102218,
                104.46127644,
                76.45994253,
                71.57052154,
                77.31629956,
                74.18979227,
                165.28073409,
                163.38737593,
                161.47917363,
                162.43740135,
                152.82760908,
                150.73006441,
                140.83429424,
                138.66509153,
                134.6973093,
                132.99813295,
                150.49072765,
                144.54019426,
                186.67331791,
                181.53060234,
                131.6530334,
                129.6622974,
                121.40843229,
                118.60957062,
                116.39092593,
                114.43649907,
                118.90182843,
                118.01810562,
                122.80554807,
                118.67236025,
                119.32751973,
                110.67782716,
                101.4901505,
                97.86646538,
                104.27922607,
                94.98366187,
                84.22986703,
                79.67752106,
                82.20284432,
                73.48358714,
                57.64971668,
                52.06401388,
                43.63114954,
                39.62299378,
                79.75655074,
                78.83538971,
                78.7233908,
                76.49070506,
                67.14054291,
                63.17929148,
                41.71139163,
                32.11614479,
                40.15115331,
                34.04905621,
                42.64541727,
                41.04761205,
                53.79855827,
                51.72007042,
                76.8045977,
                86.64375174,
                87.16152042,
                70.76570352,
                56.91432792,
                49.77622217,
                52.03441237,
                59.375,
            ],
        ]
    )

    def __init__(self):
        super().__init__(self.aal2, coords=self.aal2_centers)
        # define convenience regions - indexing from 0!!
        # self.cortex = [region for region in range(94) if region not in np.array(list(range(40, 46)) + list(range(74, 82)))]
        self.cerebellum = [region for region in range(94, 120)]
        self.thalamus = [80, 81]
        self.basal_ganglia = [region for region in range(74, 80)]
        self.amygdala = [44, 45]
        self.hippocampus = [region for region in range(40, 44)]
        self.subcortical = self.cerebellum + self.thalamus + self.basal_ganglia + self.amygdala + self.hippocampus
        self.cortex = [region for region in range(120) if region not in self.subcortical]


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

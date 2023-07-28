import os
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

import logging
from xml.etree import ElementTree
from neurolib.utils.atlases import AutomatedAnatomicalParcellation2


class LeadfieldGenerator:
    """
    A class to compute the lead-field matrix and perform related operations.

    Parameters:
    ==========
        subject (str): The subject ID.
        subjects_dir (str): Path to the directory containing the subject data.
        trans (str): Path to the transformation file.
        fs_dir (str): Path to the Freesurfer directory.

    Attributes:
    ==========
        subjects_dir (str): Path to the directory containing the subject data.
        trans (str): Path to the transformation file.
        fs_dir (str): Path to the Freesurfer directory.
        raw (mne.io.Raw): The raw EEG data.
        atlas_nii (str): Path to the NIfTI file of the atlas.
        atlas_xml (str): Path to the XML file of the atlas.

    Methods:
    =======
        load_data(raw_file, atlas_nii, atlas_xml):
            Load raw EEG data and atlas files.

        compute_leadfield(cortex_parts='full_cortex'):
            Compute the lead-field matrix based on the loaded data.

        get_labels_of_points(points, cortex_parts):
            Get labels and their corresponding codes for points in MNI space.

        downsample_leadfield_matrix(leadfield, label_codes):
            Downsample the lead-field matrix based on label codes.

        verify_leadfield(location="center", extent=10.0):
            Verify the lead-field matrix by simulating sources and computing EEG signals.

        compute_leadfield_and_save(cortex_parts='full_cortex', path_to_save=None):
            Compute the lead-field matrix and optionally save it as a binary file.

        check_atlas_missing_regions():
            Check for missing regions in the atlas based on label codes.
    """

    def __init__(self, subject):
        self.subject = subject
        self.fs_dir = None
        self.subjects_dir = None
        self.trans = None

    def __create_label_lut(self, path: str) -> dict:
        """
        Create a lookup table that contains "anatomical acronyms" corresponding to the encodings of the regions
        specified by the used anatomical atlas. Adds an empty label for code "0" if not specified otherwise by atlas.

        Parameters:
        ==========
            path (str): Path to the XML file containing label information.

        Returns:
        =======
            dict: Dictionary with keys being the integer codes of regions and the values being anatomical acronyms.

        Example:
        =======
            xml_file = 'atlas.xml'
            result = create_label_lut(xml_file)
            # Output: {'1': 'Region1', '2': 'Region2', '0': ''}
        """
        # Look up the codes ("index") and the names of the regions defined by the atlas.
        tree = ElementTree.parse(path)
        root = tree.getroot()
        label_lut = {}
        for region in root.find("data").findall("label"):
            label_lut[region.find("index").text] = region.find("name").text

        if "0 " not in label_lut.keys():
            label_lut["0"] = ""
        return label_lut

    def __get_backprojection(
        self, point_expanded: np.ndarray, affine: np.ndarray, affine_inverse: np.ndarray
    ) -> np.ndarray:
        """
        Transform MNI-mm-point into 'voxel-coordinate'.

        Parameters:
        ==========
            point_expanded (np.ndarray): First three elements are the 3D point in MNI-coordinate space (mm),
                                        last element being a 1 for the offset in transformations. `point_expanded` must have the shape of 4x1.
            affine (np.ndarray): Projects voxel-numbers to MNI coordinate space (mm). `affine` must have the shape of 4x4.
            affine_inverse (np.ndarray): Back projection from MNI space. `affine_inverse` must have the shape of 4x4.

        Returns:
        =======
            np.ndarray: The point projected back into "voxel-number-space", last element 1. Will return the shape of 4x1.

        Example:
        =======
            point = np.array([10, 20, 30, 1])
            affine_matrix = np.array([[2, 0, 0, -40], [0, 2, 0, -60], [0, 0, 2, -80], [0, 0, 0, 1]])
            inverse_affine = np.linalg.inv(affine_matrix)
            result = get_backprojection(point, affine_matrix, inverse_affine)
            # Output: array([10., 20., 30.,  1.])
        """

        # project the point from mni to voxel
        back_proj = affine_inverse @ point_expanded

        # Round to voxel resolution, multiplication with elements inverse is equivalent to division with elements of the affine here.
        back_proj_rounded = np.round(np.diag(affine_inverse) * back_proj, 0) * np.diag(affine)

        return back_proj_rounded

    def __filter_for_regions(self, label_strings: list[str], regions: list[str]) -> list[bool]:
        """
        Create a list of bools indicating if the label_strings are in the regions list.
        This function can be used if one is only interested in a subset of regions defined by an atlas.

        Parameters:
        ==========
            label_strings (list[str]): List of labels that dipoles got assigned to.
            regions (list[str]): List of strings that are the acronyms for the regions of interest.

        Returns:
        =======
            list[bool]: List of bools indicating if each label_string is in the regions list.

        Example:
        =======
            labels = ['Region1', 'Region2', 'Region3']
            regions_of_interest = ['Region1', 'Region3']
            result = filter_for_regions(labels, regions_of_interest)
            # Output: [True, False, True]
        """
        # Remark: then outside this function the label codes and label-strings can be set to nan or 0 for dipoles that are
        #  not of interest such that downsampling works smoothly.

        # in_regions = [None] * len(label_strings)
        # for idx_label, label in enumerate(label_strings):
        #     if label in regions:
        #         in_regions[idx_label] = True
        #     else:
        #         in_regions[idx_label] = False
        # return in_regions
        # Create a set for faster membership testing

        regions_set = set(regions)

        # Use list comprehension for faster filtering
        in_regions = [label in regions_set for label in label_strings]

        return in_regions

    def __get_labels_of_points(
        self,
        points: np.ndarray,
        nii_file: nib.Nifti1Image,
        xml_file: dict,
        atlas="aal2_cortical",
        cortex_parts="full_cortex",
    ) -> tuple[list[bool], np.ndarray, list[str]]:
        """
        Gives labels of regions the points fall into.

        Parameters:
        ==========
            points (np.ndarray): Nx3 array of points defined in MNI space (mm).
            nii_file (nibabel.Nifti1Image): NIfTI file representing the anatomical atlas.
            xml_file (dict): Dictionary containing "anatomical acronyms" corresponding to the encodings of the regions.
            atlas (str): Specification of the anatomical atlas. Currently only "aal2_cortical" is supported and is set as default.
            cortex_parts (str): Specification of cortex parts, defaults to "full_cortex".

        Returns:
        =======
            tuple[list[bool], np.ndarray, list[str]]: Tuple containing:
            - List of boolean values indicating if a valid assignment within the space defined by the atlas was found for each point.
            - Array of the assigned label codes for each point.
            - List of strings representing the "anatomical acronyms" of the assigned labels.

        Example:
        =======
            points = np.array([[10, 20, 30], [40, 50, 60]])
            nii_file = nib.load('atlas.nii')
            xml_file = {'1': 'Region1', '2': 'Region2', '0': ''}
            result = get_labels_of_points(points, nii_file, xml_file)
            # Output: ([True, False], array([1., 0.]), ['Region1', 'invalid'])
        """
        n_points = points.shape[0]
        label_codes = np.zeros(
            n_points
        )  # Remark: or expand points-array by one dimension and fill label-codes in there?
        label_strings = [None] * n_points
        points_found = [None] * n_points

        points_expanded = np.ones((n_points, 4))  # Expand by a column with ones only to allow for transformations
        points_expanded[:, 0:3] = points  # with affines.

        if not points.shape[1] == 3:
            raise ValueError

        # Load atlas (integer encoded volume and string-labels).
        # if atlas == "aal2" or atlas == "aal2_cortical":
        #    atlas_path = os.path.join(
        #        os.path.dirname(__file__),
        #        "../../../..",
        #        "neurolib",
        #        "data",
        #        "datasets",
        #        "aal",
        #        "atlas",
        #    )
        #    atlas_img = nib.load(os.path.join(atlas_path, "AAL2.nii"))
        #    atlas_labels_lut = create_label_lut(os.path.join(atlas_path, "AAL2.xml"))
        # else:
        #    raise ValueError("Currently only 'aal2' is supported.")
        atlas_img = nii_file
        atlas_labels_lut = xml_file

        affine = atlas_img.affine  # Transformation from voxel- to mni-space.
        affine_inverse = np.linalg.inv(affine)  # Transformation mni- to "voxel"-space.

        # Get voxel codes
        codes = atlas_img.get_fdata()
        for point_idx, point in enumerate(points_expanded):
            back_proj = self.__get_backprojection(point, affine, affine_inverse)

            try:
                label_codes[point_idx] = codes[int(back_proj[0]), int(back_proj[1]), int(back_proj[2])]

            except IndexError:
                label_codes[point_idx] = np.NAN

            if np.isnan(label_codes[point_idx]):
                points_found[point_idx] = False
                label_strings[point_idx] = "invalid"
            else:
                points_found[point_idx] = True
                label_strings[point_idx] = atlas_labels_lut[str(int(label_codes[point_idx]))]  # ToDo: clean up type-
                # conversions.
        if sum(points_found) < n_points:
            logging.error(
                f"The atlas does not specify valid labels for all the given points.\n"
                f"Total number of points: (%s) out of which (%s) were validly assigned." % (n_points, sum(points_found))
            )

        if atlas == "aal2_cortical":
            aal_2 = AutomatedAnatomicalParcellation2()
            regions = []
            k = 0

            # Select cortex part
            full_cortex = aal_2.cortex + aal_2.subcortical
            only_cortex = aal_2.cortex
            subcortical_parts = aal_2.subcortical

            if cortex_parts == "full_cortex":
                cortex_parts = full_cortex
            if cortex_parts == "only_cortex":
                cortex_parts = only_cortex
            if cortex_parts == "subcortical_parts":
                cortex_parts = subcortical_parts

            for r in cortex_parts:
                regions.append(aal_2.aal2[r + 1])
                in_regions = self.__filter_for_regions(label_strings, regions)
                k = k + 1

            for idx_point in range(len(points_found)):
                if not in_regions[idx_point]:
                    label_codes[idx_point] = 0
                    label_strings[idx_point] = ""

            print("regions number:", k)
            print("=====================================================")

        return points_found, label_codes, label_strings

    def __downsample_leadfield_matrix(
        self, leadfield: np.ndarray, label_codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Downsample the leadfield matrix by computing the average across all dipoles falling within specific regions. This process assumes a one-to-one correspondence between source positions and dipoles, as commonly found in a surface source space where the dipoles' orientations are aligned with the surface normals.

        Parameters:
        ==========
            leadfield (np.ndarray): Leadfield matrix. Channels x Dipoles.
            label_codes (np.ndarray): 1D array of region-labels assigned to the source locations.

        Returns:
        =======
            tuple[np.ndarray, np.ndarray]: Tuple containing:
            - Array that contains the label-codes of any region that at least one dipole was assigned to.
            - Channels x Regions leadfield matrix. The order of rows (channels) is unchanged compared to the input "leadfield",
            but the columns are sorted according to the "unique_labels" array.

        Example:
        =======
            leadfield = np.array([[1, 2, 3], [4, 5, 6]])
            labels = np.array([1, 0])
            result = downsample_leadfield_matrix(leadfield, labels)
            # Output: (array([1]), array([[2.5], [4.5]]))
        """
        leadfield_orig_shape = leadfield.shape
        n_channels = leadfield_orig_shape[0]

        if leadfield_orig_shape[1] != label_codes.size:
            raise ValueError(
                "The lead field matrix does not have the expected number of columns. \n"
                "Number of columns differs from labels (equal number dipoles)."
            )

        unique_labels = np.unique(label_codes)
        unique_labels = np.delete(unique_labels, np.where(np.isnan(unique_labels))[0])  # Delete NAN if present.
        # NAN would indicate point that
        # doesn't fall into space
        # covered by atlas.
        unique_labels = np.delete(unique_labels, np.where(unique_labels == 0)[0])  # Delete 0 if present. "0" in AAL2
        # is non-brain-tissue, eg. CSF.

        downsampled_leadfield = np.zeros((n_channels, unique_labels.size))

        for label_idx, label in enumerate(unique_labels):  # iterate through regions
            indices_label = np.where(label_codes == label)[0]

            downsampled_leadfield[:, label_idx] = np.mean(leadfield[:, indices_label], axis=1)

        return unique_labels, downsampled_leadfield

    def load_data(self, atlas_nii_path: str, atlas_xml_path: str):
        """
        Load raw EEG data and atlas files.

        Parameters:
        ==========
        raw_file (str): Name of the raw EEG data file.
        atlas_nii (str): Path to the NIfTI file of the atlas.
        atlas_xml (str): Path to the XML file of the atlas.
        """

        self.fs_dir = fetch_fsaverage(verbose=True)
        self.subjects_dir = os.path.dirname(self.fs_dir)

        self.trans = os.path.join(self.subjects_dir, self.subject, "bem", "fsaverage-trans.fif")

        (raw_fname,) = eegbci.load_data(subject=1, runs=[6])
        raw = mne.io.read_raw_edf(raw_fname, preload=True)

        atlas_nii_objcet = nib.load(atlas_nii_path)
        atlas_xml_object = self.__create_label_lut(atlas_xml_path)

        return raw, atlas_nii_objcet, atlas_xml_object

    def build_BEM(self, conductivity) -> str:
        """
        Create the Boundary Element Model (BEM) solution for the given subject using on the linear collocation approach.

        Parameters:
        ==========
            subject (ndarray | str): Subject identifier.
            subjects_dir (str): Subject directory path.
            fs_dir (str): FreeSurfer directory path.
            conductivity : array of int, shape (3,) or (1,). The conductivities to use for each shell. Should be a single element for a one-layer model, or three elements for a three-layer model. Defaults to ``[0.3, 0.006, 0.3]``. The MNE-C default for a single-layer model would be ``[0.3]``.


        Returns:
        =======
            mne.bem.ConductorModel: Instance of ConductorModel, that is The BEM solution.

        Example:
        =======
            subject_id = 'sample_subject'
            subjects_dir = '/path/to/subjects'
            freesurfer_dir = '/path/to/fsaverage'
            result = build_BEM(subject_id, subjects_dir, freesurfer_dir, conductivity='default')
            # Output: '/path/to/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif'
        """

        model = mne.make_bem_model(
            subject=self.subject, ico=4, conductivity=conductivity, subjects_dir=self.subjects_dir
        )
        bem = mne.make_bem_solution(model)

        return bem

    def clean_eeg_channels(self, raw: mne.io.Raw) -> None:
        """
        Clean channel names to be compatible with the standard 1020 montage.

        Parameters:
        ==========
            raw (mne.io.Raw): Raw data object.

        Returns:
        =======
            None

        Example:
        =======
            raw_data = mne.io.read_raw_edf('data.edf')
            clean_eeg_channels(raw_data)
        """
        # Clean channel names to be able to use a standard 1020 montage
        new_names = dict(
            (ch_name, ch_name.rstrip(".").upper().replace("Z", "z").replace("FP", "Fp")) for ch_name in raw.ch_names
        )
        raw.rename_channels(new_names)

    def plot_EEG_montage(self, raw: mne.io.Raw, src: mne.SourceSpaces, kind: str = "standard_1020") -> None:
        """
        Plot the EEG electrode montage on the MRI brain.

        Parameters:
        ==========
            raw (mne.io.Raw): Raw data object.
            src (mne.SourceSpaces): Source space object.
            trans (str): Path to the transformation file.
            kind (str): Type of EEG electrode layout, defaults to 'standard_1020'.

        Returns:
        =======
            None

        Example:
        =======
            raw_data = mne.io.read_raw_edf('data.edf')
            src_space = mne.setup_source_space(subject='fsaverage', subjects_dir='/path/to/subjects')
            trans_file = '/path/to/trans.fif'
            plot_EEG_montage(raw_data, src_space, trans_file)
        """

        # Read and set the EEG electrode locations, which are already in fsaverage's space (MNI space) for standard_1020:
        montage = mne.channels.make_standard_montage(kind)
        raw.set_montage(montage)
        raw.set_eeg_reference(projection=True)  # needed for inverse modeling

        # Check that the locations of EEG electrodes is correct with respect to MRI
        mne.viz.plot_alignment(
            raw.info,
            src=src,
            eeg=["original", "projected"],
            trans=self.trans,
            show_axes=True,
            mri_fiducials=True,
            dig="fiducials",
        )

    def compute_leadfield(
        self,
        raw: mne.io.Raw,
        src: mne.SourceSpaces,
        bem: mne.bem.ConductorModel,
        atlas_nii: nib.Nifti1Image,
        atlas_xml: dict,
        atlas: str = "aal2_cortical",
        cortex_parts: str = "full_cortex",
        path_to_save: str = None,
    ) -> tuple[np.ndarray, mne.Forward, np.ndarray]:
        """
        Compute the leadfield matrix.

        Parameters:
        ==========
            raw (mne.io.Raw): Raw data object.
            trans (str): Path to the transformation file.
            src (mne.SourceSpaces): Source space object.
            bem (mne.bem.ConductorModel): BEM object.
            subject (str): Subject identifier.
            atlas_nii (nibabel.Nifti1Image): NIfTI file representing the anatomical atlas.
            atlas_xml (dict): Dictionary containing "anatomical acronyms" corresponding to the encodings of the regions.
            atlas (str): Specification of the anatomical atlas, defaults to "aal2_cortical".
            cortex_parts (str): Specification of cortex parts, defaults to "full_cortex".
            path_to_save (str): Path to save the leadfield matrix as a binary file in NumPy .npy format, defaults to None.

        Returns:
        =======
            tuple[np.ndarray, mne.Forward, np.ndarray]: Tuple containing:
            - Channels x Regions leadfield matrix.
            - Forward solution object.
            - Array that contains the label-codes of any region that at least one dipole was assigned to.

        Example:
        =======
            raw_data = mne.io.read_raw_edf('data.edf')
            src_space = mne.setup_source_space(subject='fsaverage', subjects_dir='/path/to/subjects')
            trans_file = '/path/to/trans.fif'
            bem_model = mne.make_bem_model(subject='fsaverage', ico=4, conductivity='default', subjects_dir='/path/to/subjects')
            atlas_nii = nib.load('atlas.nii')
            atlas_xml = {'1': 'Region1', '2': 'Region2'}
            result = compute_leadfield(raw_data, trans_file, src_space, bem_model, 'fsaverage', atlas_nii, atlas_xml)
        """
        # Calculate the general forward solution
        ## The overall forward solution between surface source model and standard EEG montage is computed
        fwd = mne.make_forward_solution(
            raw.info, trans=self.trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
        )
        # leadfield = fwd["sol"]["data"]
        print(fwd)
        print("=====================================================")

        # Downsample the forward solution to achieve lead-field matrix
        ## With the forward solution that being calculated above, compute the average dipole value of the dipoles in each AAL atlas to acquire the lead-field matrix.

        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)

        leadfield_fixed = fwd_fixed["sol"]["data"]

        lh = fwd_fixed["src"][0]
        dip_pos_lh = np.vstack(lh["rr"][lh["vertno"]])
        rh = fwd_fixed["src"][1]
        dip_pos_rh = np.vstack(rh["rr"][rh["vertno"]])

        dip_pos = np.vstack((dip_pos_lh, dip_pos_rh))

        trans_info = mne.read_trans(self.trans)

        dip_pos_mni = mne.head_to_mni(dip_pos, subject=self.subject, mri_head_t=trans_info)

        points_found, label_codes, label_strings = self.__get_labels_of_points(
            dip_pos_mni, atlas_nii, atlas_xml, atlas=atlas, cortex_parts=cortex_parts
        )

        unique_labels, leadfield_downsampled = self.__downsample_leadfield_matrix(leadfield_fixed, label_codes)

        print(np.array(points_found).shape)
        print("=====================================================")

        print("Leadfield size : %d sensors x %d dipoles" % leadfield_downsampled.shape)
        print("=====================================================")

        print(leadfield_downsampled)
        print("=====================================================")
        # Export the leadfield matrix an array to a binary file in NumPy .npy format.
        if path_to_save is not None:
            np.save(os.path.join(path_to_save, "leadfield_downsampled"), leadfield_downsampled)
            print(f"The leadfiled matrix is saved as a binary file in NumPy .npy format at {path_to_save}")
            print("=====================================================")

        return leadfield_downsampled, fwd, unique_labels

    def check_atlas_missing_regions(self, atlas_xml_object, unique_labels: np.ndarray) -> None:
        """
        Investigate the missing regions of the atlas.

        Parameters:
        ==========
            atlas_xml_path (str): Path to the XML file containing label information.
            unique_labels (np.ndarray): Array containing the label-codes of any region that at least one dipole was assigned to.

        Returns:
        =======
            None

        Example:
        =======
            xml_file = 'atlas.xml'
            labels = np.array([1, 2, 3])
            result = check_atlas_missing_regions(xml_file, labels)
        """
        aal_2 = AutomatedAnatomicalParcellation2()
        regions = []
        k = 0
        full_cortex = aal_2.cortex + aal_2.subcortical
        for r in full_cortex:
            regions.append(aal_2.aal2[r + 1])
            k = k + 1

        print("total region number:", k)
        print("=====================================================")

        label_numbers = np.array(list(map(int, atlas_xml_object.keys())))[:-1]  # Convert the keys to integers
        missed_region_labels = np.setdiff1d(label_numbers, unique_labels)
        print("missed region labels:", missed_region_labels)
        print("=====================================================")

        missed_region_labels_str = missed_region_labels.astype(str)
        # missed_region_labels_str = np.core.defchararray.add(missed_region_labels.astype(str), '')
        missed_region_values = list(
            atlas_xml_object[label] for label in missed_region_labels_str if label in atlas_xml_object
        )
        print("missed region names:", missed_region_values)
        print("=====================================================")

        subset = set(missed_region_labels)
        missed_region_index = np.array([i + 1 for i, e in enumerate(label_numbers) if e in subset])
        print("missed region index:", missed_region_index)
        print("=====================================================")

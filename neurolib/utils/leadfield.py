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
    Authors: Mohammad Orabe <orabe.mhd@gmail.com>
             Zixuan liu <zixuan.liu@campus.tu-berlin.de> 

    A class to compute the lead-field matrix and perform related operations.
    The default loaded data is the template data 'fsaverage'.
    To establish an AAL2 atlas source space, the average dipole value within each atlas annotation is computed, a process referred to as downsampling.
    The initial step is to generate the surface source model.
    The downsampling process need NIfTI file and XML file of the AAL2 atlas.


    Parameters:
    ==========
        fs_dir (str): Path to the downloaded 'fsaverage' directory, set as default data.
        subject (str): The name of the subject.
        subjects_dir (str): Path to the directory containing the subject data.
        trans (str): Path to the coregistration transformation file.
        atlas_nii (str): Path to the NIfTI file of the atlas.
        atlas_xml (str): Path to the XML file of the atlas.

    Attributes:
    ==========
        raw (mne.io.Raw): The raw EEG data.


    Methods:
    =======
        load_data(subject, subjects_dir):
            Load subject data and its directory, 'fsaverage' is set as default. For user-specific data, a coregistration transformation file needed to be generated.

        load_transformation file(trans):
            load the transformation file of the subject, 'fsaverage' has default transformation file.

        build_BEM(subject, conductivity, subjects_dir)
            Construct BEM for the given subject head model.

        generate_surface_source_space(subject, spacing, add_dist):
            Generate the overall surface source model.

        EEG_coregistration(subject, configuration, src, trans, visualization):
            Align the selected EEG configuration with the subject and visualization.

        calculate_general_forward_solution(raw, trans, src, bem, eeg, mindist, n_jobs):
            Compute the general forward solution based on given subject, BEM, and EEG configuration.

        downsample_leadfield_matrix(leadfield, label_codes, atlas_nii, atlas_xml):
            Downsample the lead-field matrix according to AAL2 atlas based on general forward solution.

        check_atlas_missing_regions():
            Check for missing regions in the atlas based on label codes.
    """

    def __init__(self, subject):
        self.subject = subject
        self.fs_dir = None
        self.subjects_dir = None
        self.trans = None

    def load_data(self, subjects_dir=None, subject="fsaverage"):
        """
        Load subject data.

        Parameters:
        ==========
        subject (str): The name of the subject, default set as 'fsaverage'.
        subjects_dir (str): The directory of the subject.

        """
        if subject == "fsaverage":
            # Download the template data 'fsaverage'
            self.fs_dir = fetch_fsaverage(verbose=True)
            self.subjects_dir = os.path.dirname(self.fs_dir)
            print("Load template data 'fsaverage'")
        else:
            self.subjects_dir = subjects_dir
            # Generate transformation file, detail see https://mne.tools/stable/generated/mne.gui.coregistration.html#mne.gui.coregistration
            mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)

        # (raw_fname,) = eegbci.load_data(subject=1, runs=[6])
        # raw = mne.io.read_raw_edf(raw_fname, preload=True)

    def load_transformation_file(self, trans_path, subject="fsaverage"):
        """
        Load transformation file.

        Parameters:
        ==========
        trans_path (str): The directory of the transformation file

        """
        # Load the generated transformation file
        if subject == "fsaverage":
            self.trans = os.path.join(self.subjects_dir, self.subject, "bem", "fsaverage-trans.fif")
            print("Load default transformation file of 'fsaverage'")
        else:
            self.trans = trans_path

    def build_BEM(
        self,
        conductivity=(0.3, 0.006, 0.3),
        visualization=True,
        brain_surfaces="white",
        orientation="coronal",
        slices=[50, 100, 150, 200],
    ):
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
            mne.bem.ConductorModel: BEM of the given head model.
            plot_bem_kwargs: Image information of the given mri data

        """

        model = mne.make_bem_model(
            subject=self.subject,
            ico=4,
            conductivity=conductivity,
            subjects_dir=self.subjects_dir,
        )
        bem = mne.make_bem_solution(model)

        # Visualization of the BEM
        plot_bem_kwargs = dict(
            subject=self.subject,
            subjects_dir=self.subjects_dir,
            brain_surfaces=brain_surfaces,
            orientation=orientation,
            slices=slices,
        )

        if visualization == True:
            mne.viz.plot_bem(**plot_bem_kwargs)

        return bem, plot_bem_kwargs

    def generate_surface_source_space(self, plot_bem_kwargs, spacing="ico4", add_dist="patch", visualization=True):
        """
        Generate the overall surface source model.

        Parameters:
        ==========
            subject (ndarray | str): Subject identifier.
            subjects_dir (str): Subject directory path.
            spacing (str) : The spacing to use. Can be 'ico#' for a recursively subdivided icosahedron, 'oct#' for a recursively subdivided octahedron, 'all' for all points, or an integer to use approximate distance-based spacing (in mm).
            add_dist (bool | str): Add distance and patch information to the source space.

        Returns:
        =======
            src (mne.SourceSpaces): Surface source space object.

        """

        if self.subject == "fsaverage":
            src = os.path.join(self.fs_dir, "bem", "fsaverage-ico-5-src.fif")
        else:
            src = mne.setup_source_space(
                subject=self.subject,
                spacing=spacing,
                add_dist=add_dist,
                subjects_dir=self.subjects_dir,
            )

        if visualization == True:
            mne.viz.plot_bem(src=src, **plot_bem_kwargs)

        return src

    def EEG_coregistration(self, src, configuration="standard_1020", visualization=True):
        """
        Align the selected EEG configuration with the subject and visualization.

        Parameters:
        ==========
            src (mne.SourceSpaces): Source space object.
            trans (str): Path to the transformation file.
            configuration (str): Type of EEG electrode layout, defaults to 'standard_1020'.

        Returns:
        =======
            raw (mne.io.Raw): Raw data coregistrated with EEG.
        """

        # Load the EEGBCI data
        (raw_fname,) = eegbci.load_data(subject=1, runs=[6])
        raw = mne.io.read_raw_edf(raw_fname, preload=True)

        # Clean channel names to be able to use a standard 1020 montage
        new_names = dict(
            (ch_name, ch_name.rstrip(".").upper().replace("Z", "z").replace("FP", "Fp")) for ch_name in raw.ch_names
        )
        raw.rename_channels(new_names)

        # Read and set the EEG electrode locations, which are already in fsaverage's space (MNI space) for standard_1020:
        montage = mne.channels.make_standard_montage(configuration)
        raw.set_montage(montage)
        raw.set_eeg_reference(projection=True)  # needed for inverse modeling

        # Check that the locations of EEG electrodes is correct with respect to MRI
        if visualization == True:
            mne.viz.plot_alignment(
                raw.info,
                src=src,
                eeg=["original", "projected"],
                trans=self.trans,
                show_axes=True,
                mri_fiducials=True,
                dig="fiducials",
            )

        return raw

    def calculate_general_forward_solution(self, raw, src, bem, eeg=True, mindist=5.0):
        """
        Calculate the general forward solution

        Parameters:
        ==========
            raw (mne.io.Raw): Raw data coregistrated with EEG.
            src (mne.SourceSpaces): Surface source space object.
            trans (str): Path to the transformation file.
            bem (mne.bem.ConductorModel): BEM of the given head model.

        Returns:
        =======
            fwd: The general forward solution.

        """

        # Computer the general forward solution
        fwd = mne.make_forward_solution(
            raw.info,
            trans=self.trans,
            src=src,
            bem=bem,
            eeg=eeg,
            mindist=mindist,
            n_jobs=None,
        )
        print("The general forward solution:", fwd)
        print("=====================================================")

        return fwd

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

        """
        # Remark: then outside this function the label codes and label-strings can be set to nan or 0 for dipoles that are not of interest such that downsampling works smoothly.

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
        cortex_parts="only_cortical_parts",
    ) -> tuple[list[bool], np.ndarray, list[str]]:
        """
        Gives labels of regions the points fall into.

        Parameters:
        ==========
            points (np.ndarray): Nx3 array of points defined in MNI space (mm).
            nii_file (nibabel.Nifti1Image): NIfTI file representing the anatomical atlas.
            xml_file (dict): Dictionary containing "anatomical acronyms" corresponding to the encodings of the regions.
            atlas (str): Specification of the anatomical atlas. Currently only "aal2_cortical" is supported and is set as default.
            cortex_parts (str): Specification of cortex parts, defaults to "only_cortical_parts".

        Returns:
        =======
            tuple[list[bool], np.ndarray, list[str]]: Tuple containing:
            - List of boolean values indicating if a valid assignment within the space defined by the atlas was found for each point.
            - Array of the assigned label codes for each point.
            - List of strings representing the "anatomical acronyms" of the assigned labels.

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
                label_strings[point_idx] = atlas_labels_lut[
                    str(int(label_codes[point_idx]))
                ]  # ToDo: clean up type- conversions.
        if sum(points_found) < n_points:
            logging.error(
                f"The atlas does not specify valid labels for all the given points.\n"
                f"Total number of points: (%s) out of which (%s) were validly assigned." % (n_points, sum(points_found))
            )

        if atlas == "aal2_cortical":
            aal_2 = AutomatedAnatomicalParcellation2()
            regions = []

            # Select cortex part
            full_cortex = aal_2.cortex + aal_2.subcortical
            only_cortical_parts = aal_2.cortex
            subcortical_parts = aal_2.subcortical

            if cortex_parts == "full_cortex":
                cortex_parts = full_cortex
            if cortex_parts == "only_cortical_parts":
                cortex_parts = only_cortical_parts
            if cortex_parts == "subcortical_parts":
                cortex_parts = subcortical_parts

            for r in cortex_parts:
                regions.append(aal_2.aal2[r + 1])
                in_regions = self.__filter_for_regions(label_strings, regions)

            for idx_point in range(len(points_found)):
                if not in_regions[idx_point]:
                    label_codes[idx_point] = 0
                    label_strings[idx_point] = ""

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
        # NAN would indicate point that doesn't fall into space covered by atlas.
        unique_labels = np.delete(
            unique_labels, np.where(unique_labels == 0)[0]
        )  # Delete 0 if present. "0" in AAL2 is non-brain-tissue, eg. CSF.

        downsampled_leadfield = np.zeros((n_channels, unique_labels.size))

        for label_idx, label in enumerate(unique_labels):  # iterate through regions
            indices_label = np.where(label_codes == label)[0]

            downsampled_leadfield[:, label_idx] = np.mean(leadfield[:, indices_label], axis=1)

        return unique_labels, downsampled_leadfield

    def compute_downsampled_leadfield(
        self,
        fwd,
        atlas_nii_path,
        atlas_xml_path,
        atlas="aal2_cortical",
        cortex_parts="only_cortical_parts",
        path_to_save=None,
    ):
        """
        Compute the leadfield matrix.

        Parameters:
        ==========
            raw (mne.io.Raw): Raw data object.
            trans (str): Path to the transformation file.
            src (mne.SourceSpaces): Source space object.
            bem (mne.bem.ConductorModel): BEM object.
            subject (str): Subject identifier.
            atlas_nii_path (str): Path to the NIfTI file of the atlas.
            atlas_xml_path (str): Path to the XML file of the atlas.
            atlas (str): Specification of the anatomical atlas, defaults to "aal2_cortical".
            cortex_parts (str): Specification of cortex parts, defaults to "only_cortical_parts".
            path_to_save (str): Path to save the leadfield matrix as a binary file in NumPy .npy format, defaults to None.

        Returns:
        =======
            tuple[np.ndarray, mne.Forward, np.ndarray]: Tuple containing:
            - Channels x Regions leadfield matrix.
            - Forward solution object.
            - Array that contains the label-codes of any region that at least one dipole was assigned to.

        """
        # Calculate the general forward solution

        # Downsample the forward solution to achieve lead-field matrix
        ## With the forward solution that being calculated above, compute the average dipole value of the dipoles in each AAL atlas to acquire the lead-field matrix.

        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)

        leadfield_fixed = fwd_fixed["sol"]["data"]

        atlas_nii_file = nib.load(atlas_nii_path)

        atlas_xml_file = self.__create_label_lut(atlas_xml_path)

        lh = fwd_fixed["src"][0]
        dip_pos_lh = np.vstack(lh["rr"][lh["vertno"]])
        rh = fwd_fixed["src"][1]
        dip_pos_rh = np.vstack(rh["rr"][rh["vertno"]])

        dip_pos = np.vstack((dip_pos_lh, dip_pos_rh))

        trans_info = mne.read_trans(self.trans)

        dip_pos_mni = mne.head_to_mni(dip_pos, subject=self.subject, mri_head_t=trans_info)

        points_found, label_codes, label_strings = self.__get_labels_of_points(
            dip_pos_mni,
            atlas_nii_file,
            atlas_xml_file,
            atlas=atlas,
            cortex_parts=cortex_parts,
        )

        unique_labels, leadfield_downsampled = self.__downsample_leadfield_matrix(leadfield_fixed, label_codes)

        print("Lead-field matrix's size : %d sensors x %d dipoles" % leadfield_downsampled.shape)
        print("=====================================================")

        print("Downsampled lead-field matrix:", leadfield_downsampled)
        print("=====================================================")
        # Export the leadfield matrix an array to a binary file in NumPy .npy format.
        if path_to_save is not None:
            np.save(
                os.path.join(path_to_save, "leadfield_downsampled"),
                leadfield_downsampled,
            )
            print(f"The leadfiled matrix is saved as a binary file in NumPy .npy format at {path_to_save}")
            print("=====================================================")

        return leadfield_downsampled, unique_labels

    def check_atlas_missing_regions(self, atlas_xml_path, unique_labels):
        """
        Investigate the missing regions of the atlas.

        Parameters:
        ==========
            atlas_xml_path (str): Path to the XML file containing label information.
            unique_labels (np.ndarray): Array containing the label-codes of any region that at least one dipole was assigned to.

        Returns:
        =======
            None

        """

        aal_2 = AutomatedAnatomicalParcellation2()
        full_cortex = aal_2.cortex + aal_2.subcortical
        total_region_quantity = np.array(full_cortex).shape[0]
        missed_region_quantity = np.array(full_cortex).shape[0] - np.array(unique_labels).shape[0]

        print("total region quantity:", total_region_quantity)
        print("missed region quantity: ", missed_region_quantity)
        print("=====================================================")

        atlas_xml_file = self.__create_label_lut(atlas_xml_path)

        label_numbers = np.array(list(map(int, atlas_xml_file.keys())))[:-1]  # Convert the keys to integers
        missed_region_labels = np.setdiff1d(label_numbers, unique_labels)
        print("missed region labels:", missed_region_labels)
        print("=====================================================")

        missed_region_labels_str = missed_region_labels.astype(str)
        # missed_region_labels_str = np.core.defchararray.add(missed_region_labels.astype(str), '')
        missed_region_values = list(
            atlas_xml_file[label] for label in missed_region_labels_str if label in atlas_xml_file
        )
        print("missed region names:", missed_region_values)
        print("=====================================================")

        subset = set(missed_region_labels)
        missed_region_indices = np.array([i + 1 for i, e in enumerate(label_numbers) if e in subset])
        print("missed region indices:", missed_region_indices)
        print("=====================================================")

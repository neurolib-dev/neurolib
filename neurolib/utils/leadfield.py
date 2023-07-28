# leadfield_functions.py
# Description: Contains functions for leadfield generation.
# Author: Mohammad Orabe
#         ZiXuan Liu
#         Nikola Jajcay
#
# Contact:
#          orabe.mhd@gmail.com
#          a837707601@gmail.com
#          jajcay@ni.tu-berlin.de
#          martink@bccn-berlin.de

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


def filter_for_regions(label_strings: list[str], regions: list[str]) -> list[bool]:
    """Create a list of bools indicating if the label_strings are in the regions list.
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


def create_label_lut(path: str) -> dict:
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


def get_backprojection(point_expanded: np.ndarray, affine: np.ndarray, affine_inverse: np.ndarray) -> np.ndarray:
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


def get_labels_of_points(
    points: np.ndarray, nii_file: nib.Nifti1Image, xml_file: dict, atlas="aal2_cortical", cortex_parts="full_cortex"
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
    label_codes = np.zeros(n_points)  # Remark: or expand points-array by one dimension and fill label-codes in there?
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
        back_proj = get_backprojection(point, affine, affine_inverse)

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
            in_regions = filter_for_regions(label_strings, regions)
            k = k + 1

        for idx_point in range(len(points_found)):
            if not in_regions[idx_point]:
                label_codes[idx_point] = 0
                label_strings[idx_point] = ""

        print("regions number:", k)
        print("=====================================================")

    return points_found, label_codes, label_strings


def downsample_leadfield_matrix(leadfield: np.ndarray, label_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def load_data(atlas_nii_path: str, atlas_xml_path: str) -> tuple[str, str, str, mne.io.Raw, nib.nifti1.Nifti1Image, dict]:
    """
    Download and load the data needed for leadfield generation.

    Parameters:
    ==========
        atlas_nii_path (str): Path to the NIfTI file containing the anatomical atlas.
        atlas_xml_path (str): Path to the XML file containing label information.

    Returns:
    =======
        tuple[str, str, str, mne.io.Raw, nib.Nifti1Image, dict]: Tuple containing:
        - Subject directory path.
        - Path to the transformation file.
        - FreeSurfer directory path.
        - Raw data.
        - NIfTI file representing the anatomical atlas.
        - Dictionary containing "anatomical acronyms" corresponding to the encodings of the regions.

    Example:
    =======
        atlas_nii = 'atlas.nii'
        atlas_xml = 'atlas.xml'
        result = load_data(atlas_nii, atlas_xml)
        # Output: ('/path/to/subjects_dir', '/path/to/trans.fif', '/path/to/fsaverage', raw_data, atlas_nii_data, {'1': 'Region1', '2': 'Region2'})
    """

    ## Download the template MRI "fsaverage", user-specific data can replace the dataset.
    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)

    trans = os.path.join(
        subjects_dir, "fsaverage", "bem", "fsaverage-trans.fif"
    )  # MNE has a built-in fsaverage transformation

    # Load and coregistrate standard EEG configuration
    ## The standard 1020 EEG electrode locations are already calculated in fsaverage's space (MNI space)
    # Load the data
    (raw_fname,) = eegbci.load_data(subject=1, runs=[6])
    raw = mne.io.read_raw_edf(raw_fname, preload=True)

    atlas_nii = nib.load(atlas_nii_path)
    atlas_xml = create_label_lut(atlas_xml_path)

    return subjects_dir, trans, fs_dir, raw, atlas_nii, atlas_xml


def build_BEM(subject: str, subjects_dir: str, fs_dir: str, conductivity: str = "default_conductivity") -> str:
    """
    Create the Boundary Element Model (BEM) solution for the given subject using on the linear collocation approach.

    Parameters:
    ==========
        subject (str): Subject identifier.
        subjects_dir (str): Subject directory path.
        fs_dir (str): FreeSurfer directory path.
        conductivity (str): Conductivity specification, defaults to "default_conductivity".

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

    # TODO: test this condition!
    if conductivity == "default_conductivity":
        # Default BEM for "fsaverage"
        bem = os.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    # Manually build up BEM
    else:
        model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)

    return bem


def clean_eeg_channels(raw: mne.io.Raw) -> None:
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

    # # Initialize an empty dictionary to store the new channel names
    # new_names = {}

    # # Loop through each channel name in raw.ch_names
    # for ch_name in raw.ch_names:
    #     # Apply the required transformations
    #     new_name = ch_name.rstrip(".").upper().replace("Z", "z").replace("FP", "Fp")

    #     # Add the transformed channel name to the new_names dictionary
    #     new_names[ch_name] = new_name

    # # Rename the channels in the raw object with the new_names dictionary
    # raw.rename_channels(new_names)


def plot_EEG_montage(raw: mne.io.Raw, src: mne.SourceSpaces, trans: str, kind: str = "standard_1020") -> None:
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
        trans=trans,
        show_axes=True,
        mri_fiducials=True,
        dig="fiducials",
    )
    # plt.show()


def compute_leadfield(
    raw: mne.io.Raw,
    trans: str,
    src: mne.SourceSpaces,
    bem: mne.bem.ConductorModel,
    subject: str,
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
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None)
    leadfield = fwd["sol"]["data"]
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

    trans_info = mne.read_trans(trans)

    dip_pos_mni = mne.head_to_mni(dip_pos, subject=subject, mri_head_t=trans_info)

    points_found, label_codes, label_strings = get_labels_of_points(
        dip_pos_mni, atlas_nii, atlas_xml, atlas=atlas, cortex_parts=cortex_parts
    )

    unique_labels, leadfield_downsampled = downsample_leadfield_matrix(leadfield_fixed, label_codes)

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


def check_atlas_missing_regions(atlas_xml_path: str, unique_labels: np.ndarray) -> None:
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

    xml_file = create_label_lut(atlas_xml_path)
    label_numbers = np.array(list(map(int, xml_file.keys())))[:-1]  # Convert the keys to integers
    missed_region_labels = np.setdiff1d(label_numbers, unique_labels)
    print("missed region labels:", missed_region_labels)
    print("=====================================================")

    missed_region_labels_str = missed_region_labels.astype(str)
    # missed_region_labels_str = np.core.defchararray.add(missed_region_labels.astype(str), '')
    missed_region_values = list(xml_file[label] for label in missed_region_labels_str if label in xml_file)
    print("missed region names:", missed_region_values)
    print("=====================================================")

    subset = set(missed_region_labels)
    missed_region_index = np.array([i + 1 for i, e in enumerate(label_numbers) if e in subset])
    print("missed region index:", missed_region_index)
    print("=====================================================")


def verify_leadfield(
    subject: str, subjects_dir: str, fwd: mne.Forward, location: str = "center", extent: float = 10.0
) -> None:
    """
    Verification of lead-field matrix.

    Parameters:
    ==========
        subject (str): Subject identifier.
        subjects_dir (str): Subject directory path.
        fwd (mne.Forward): Forward solution object.
        location (str): Region within the brain, defaults to "center".
        extent (float): Extent in mm of the region, defaults to 10.0.

    Returns:
    =======
        None

    Example:
    =======
        subject_id = 'sample_subject'
        subjects_dir = '/path/to/subjects'
        fwd_solution = mne.make_forward_solution(...)
        verify_leadfield(subject_id, subjects_dir, fwd_solution)
    """
    # TODO: Test this function
    # Verification of lead-field matrix
    # To simulate sources, we need a source space.
    src = fwd["src"]

    # To select a region to activate, we use the caudal middle frontal to grow a region of interest.
    selected_label = mne.read_labels_from_annot(subject, regexp="caudalmiddlefrontal-lh", subjects_dir=subjects_dir)[0]
    # location="center" # Use the center of the region as a seed.
    # extent = 10.0  # Extent in mm of the region.
    label = mne.label.select_sources(
        subject, selected_label, location=location, extent=extent, subjects_dir=subjects_dir
    )

    # source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
    # source_simulator.add_data(label, source_time_series, events)

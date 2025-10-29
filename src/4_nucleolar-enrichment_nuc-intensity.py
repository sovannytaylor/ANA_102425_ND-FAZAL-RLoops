"""Detect puncta, measure features, visualize data
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import functools
import cv2
from skimage import measure, segmentation, morphology
from scipy.stats import skewtest, skew
from skimage.morphology import remove_small_objects
from statannotations.Annotator import Annotator
from loguru import logger
from matplotlib_scalebar.scalebar import ScaleBar
plt.rcParams.update({'font.size': 14})

input_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/napari_masking/'
output_folder = 'python_results/summary_calculations/'
plotting_folder = 'python_results/plotting/'
proof_folder = 'python_results/proofs/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

# FIX: this block previously checked plotting_folder twice; now creates proof_folder
if not os.path.exists(proof_folder):
    os.mkdir(proof_folder)

# ---------- helpers for proof panels ----------

AUTOSCALE_PROOFS = False  # 👈 Toggle: True for contrast-enhanced, False for absolute 0–65535 scaling

def _to_uint8(img, vmin=None, vmax=None, autoscale=AUTOSCALE_PROOFS):
    """
    Convert image to uint8 for display.

    Parameters
    ----------
    img : np.ndarray
        Input image (any numeric type).
    vmin, vmax : float, optional
        Explicit intensity bounds. If None, defaults depend on autoscale.
    autoscale : bool
        If True, rescales per image using 1–99th percentile range.
        If False, uses absolute scaling (default 0–65535 for 16-bit images).

    Returns
    -------
    np.ndarray (uint8)
        Image scaled to 0–255 range.
    """
    img = img.astype(np.float32)

    if autoscale:
        # Percentile-based local normalization
        if vmin is None or vmax is None:
            vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
            if vmin == vmax:
                vmin, vmax = float(img.min()), float(img.max()) if img.max() != img.min() else (0.0, 1.0)
        img = np.clip((img - vmin) / (vmax - vmin + 1e-9), 0, 1)
    else:
        # Absolute scaling (e.g. 16-bit range)
        vmin = 0 if vmin is None else vmin
        vmax = 65535 if vmax is None else vmax
        img = np.clip((img - vmin) / (vmax - vmin + 1e-9), 0, 1)

    return (img * 255).astype(np.uint8)

def _label_outline(mask_bool):
    """Return boolean outline of a (binary) mask."""
    return segmentation.find_boundaries(mask_bool, mode='outer')

def _make_nucleolar_mask_for_proof(ch0, nuc_mask_bool):
    """
    Build a simple binary nucleolar mask for visualization purposes:
    threshold ch0 using 2*std within nuclei, remove tiny objects.
    """
    nuc_vals = ch0[nuc_mask_bool]
    if nuc_vals.size == 0:
        return np.zeros_like(ch0, dtype=np.uint8)
    thr = 2.0 * np.std(nuc_vals)
    bin_img = (ch0 > thr) & nuc_mask_bool
    lab = measure.label(bin_img.astype(np.uint8))
    lab = remove_small_objects(lab, 9)
    return (lab > 0).astype(np.uint8)

def save_proof_panel(name, img_stack, out_dir):
    """
    img_stack: (4, H, W) = [ch0, ch1, ch2, nuc_mask_filtered]
    Saves a 1x4 panel PNG: ch0, ch1, nucleolar_mask (binary), ch2 + outlines.
    """
    ch0 = img_stack[0]
    ch1 = img_stack[1]
    ch2 = img_stack[2]
    nuc_mask = img_stack[3]  # filtered nuclear mask (labels or binary)

    nuc_mask_bool = (nuc_mask > 0)
    nucleolar_mask = _make_nucleolar_mask_for_proof(ch0, nuc_mask_bool)
    nuc_outline = _label_outline(nuc_mask_bool)

    # Convert to uint8 display images (respecting global autoscale flag)
    ch0_u8 = _to_uint8(ch0, autoscale=AUTOSCALE_PROOFS)
    ch1_u8 = _to_uint8(ch1, autoscale=AUTOSCALE_PROOFS)
    ch2_u8 = _to_uint8(ch2, autoscale=AUTOSCALE_PROOFS)

    # Create RGB for channel 2 to overlay outlines (red)
    ch2_rgb = np.dstack([ch2_u8, ch2_u8, ch2_u8]).copy()
    ch2_rgb[nuc_outline] = np.array([255, 0, 0], dtype=np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(ch0_u8, cmap='gray')
    axes[0].set_title('Channel 0')
    axes[1].imshow(ch1_u8, cmap='gray')
    axes[1].set_title('Channel 1')
    axes[2].imshow(nucleolar_mask, cmap='gray')
    axes[2].set_title('Nucleolar Mask')
    axes[3].imshow(ch2_rgb)
    axes[3].set_title('Channel 2 + Mask Outlines')
    for ax in axes:
        ax.axis('off')

    out_path = os.path.join(out_dir, f"{name}_proof.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------- end proof helpers ----------


def feature_extractor(mask, properties=False):
    if not properties:
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter', 'coords']
    return pd.DataFrame(skimage.measure.regionprops_table(mask, properties=properties))


# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

masks = {masks.replace('_mask.npy', ''): np.load(
    f'{mask_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{mask_folder}') if '_mask.npy' in masks}

# Assumes images[key] shape is (C, H, W) and masks[key] shape is (1, H, W) or (H, W)
image_mask_dict = {
    key: np.stack([
        images[key][0],  # Channel 0
        images[key][1],  # Channel 1
        masks[key][0] if masks[key].ndim == 3 else masks[key]  # Handle 2D or 3D mask
    ])
    for key in masks
}

# ----------------collect feature information----------------
# remove saturated nucs in case some were added during manual validation
not_saturated = {}
structure_element = np.ones((16, 16)).astype(int)

for name, image in image_mask_dict.items():
    labels_filtered = []
    unique_val, counts = np.unique(image[-1, :, :], return_counts=True)

    num_cells_before = len(unique_val) - 1  # exclude background
    print(f"{name}: Cells before filter = {num_cells_before}")

    # loop to remove saturated nucs
    for label in unique_val[1:]:
        pixel_count = np.count_nonzero(image[-1, :, :] == label)
        nuc_mask = np.where(image[-1, :, :] == label, label, 0)
        nuc_eroded = morphology.erosion(nuc_mask, structure_element)
        ch0_nuc = np.where(nuc_eroded == label, image[1, :, :], 0)
        ch0_nuc_saturated_count = np.count_nonzero(nuc_eroded == 65535)
        if ((ch0_nuc_saturated_count / pixel_count) < 0.05):
            labels_filtered.append(nuc_eroded)

    nucs_filtered = np.sum(labels_filtered, axis=0)
    num_cells_after = len(labels_filtered)
    print(f"{name}: Cells after filter = {num_cells_after}")

    nucs_filtered_stack = np.stack(
        (image[0, :, :], image[1, :, :], image[2, :, :], nucs_filtered))
    not_saturated[name] = nucs_filtered_stack

# ---------- generate proof panels ----------
for name, img_stack in not_saturated.items():
    save_proof_panel(name, img_stack, proof_folder)
print(f"Saved proof panels to: {proof_folder}")


## =-


# now collect nucleolus masks and nuc features info
logger.info('collecting feature info')
feature_information_list = []
for name, image in not_saturated.items():
    # logger.info(f'Processing {name}')
    labels_filtered = []
    unique_val, counts = np.unique(image[-1, :, :], return_counts=True)

    # find nuc outlines for later plotting
    nuc_binary_mask = np.where(image[-1, :, :] != 0, 1, 0)
    contours = measure.find_contours(nuc_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]

    # loop to extract params from cells
    for num in unique_val[1:]:
        # last channel (-1) is always the mask - in this case, the nuclear mask
        nuc = np.where(image[-1, :, :] == num, image[-1, :, :], 0)
        # channel 1 = peptide intensity
        pepchan = np.where(image[-1, :, :] == num, image[1, :, :], 0)
        pepchan_mean = np.mean(pepchan[pepchan != 0])

        # channel 0 = nucleolus mask proxy channel
        nucleoluschan = np.where(image[-1, :, :] == num, image[0, :, :], 0)
        nucleoluschan_std = np.std(nucleoluschan[nucleoluschan != 0])
        nucleoluschan_mean = np.mean(nucleoluschan[nucleoluschan != 0])

        # simple threshold; you can refine later
        binary = (nucleoluschan > ((nucleoluschan_std * 2))).astype(int)
        nucleolar_masks = measure.label(binary)
        nucleolar_masks = remove_small_objects(nucleolar_masks, 9)

        # measure properties of nucleolar masks
        nucleolus_properties = feature_extractor(nucleolar_masks).add_prefix('nucleolar_')

        # per-nucleolus peptide metrics
        peptide_nucleol_cv_list = []
        peptide_nucleol_skew_list = []
        peptide_nucleol_intensity_list = []
        for peptide_nucleol_num in np.unique(nucleolar_masks)[1:]:
            # per nucleolus peptide intensity values
            peptide_nucleol = np.where(nucleolar_masks == peptide_nucleol_num, image[1, :, :], 0)
            peptide_nucleol = peptide_nucleol[peptide_nucleol != 0]

            peptide_nucleol_cv = np.std(peptide_nucleol) / np.mean(peptide_nucleol)
            peptide_nucleol_cv_list.append(peptide_nucleol_cv)
            peptide_nucleol_skew_list.append(skew(peptide_nucleol))
            peptide_nucleol_intensity_list.append(np.mean(peptide_nucleol))

        # store measurements
        nucleolus_properties['peptide_nucleol_cv'] = peptide_nucleol_cv_list
        nucleolus_properties['peptide_nucleol_skew'] = peptide_nucleol_skew_list
        nucleolus_properties['peptide_nucleol_intensity'] = peptide_nucleol_intensity_list

        # if no nucleoli, create one zero row so the cell is still represented
        if len(nucleolus_properties) < 1:
            nucleolus_properties.loc[len(nucleolus_properties)] = 0

        # make df and add nuc and image info
        properties = pd.concat([nucleolus_properties])
        properties['image_name'] = name
        properties['nuc_number'] = num

        # nucleus geometry/intensity (per cell, repeated on each nucleolus row)
        properties['nuc_size'] = np.size(nuc[nuc != 0])               # pixel area
        properties['nuc_intensity_mean'] = pepchan_mean               # mean peptide intensity in nucleus

        # add nuc outlines to coords
        properties['nuc_coords'] = [contour] * len(properties)

        feature_information_list.append(properties)

feature_information = pd.concat(feature_information_list)
logger.info('completed feature collection')

# adding columns based on image_name
feature_information['condition'] = feature_information['image_name'].str.split('_').str[1]
feature_information['rep'] = feature_information['image_name'].str.extract(r'_(\d+)-\d+$')

# add aspect ratio and circularity
feature_information['peptide_nucleol_aspect_ratio'] = (
    feature_information['nucleolar_minor_axis_length'] / feature_information['nucleolar_major_axis_length']
)
feature_information['peptide_nucleol_circularity'] = (
    (12.566 * feature_information['nucleolar_area']) / (feature_information['nucleolar_perimeter'] ** 2)
)

# ===================== NEW: Correct cell-level enrichment =====================
# Sum peptide signal inside ALL nucleoli per cell, then divide by total nuclear signal.

# 1) total nucleolar peptide signal per cell (sum of per-nucleolus means × area is better,
#     but you currently have per-nucleolus mean only. If you later add per-nucleolus area,
#     replace the sum below with sum(mean * area). For now we assume mean over nucleolar pixels
#     was computed; if you want *total* signal, prefer computing and summing sums, not means.)
# If your 'feature_extractor' also gives 'nucleolar_area', you can compute per-nucleolus total:
# per_nucleolus_total = peptide_nucleol_mean * nucleolar_area
# and then sum that. The block below handles both cases.

if 'nucleolar_area' in feature_information.columns:
    # estimate total nucleolar peptide signal = sum(mean * area) across nucleoli
    feature_information['per_nucleolus_total_signal'] = (
        feature_information['peptide_nucleol_intensity'] * feature_information['nucleolar_area']
    )
    agg_signal_col = 'per_nucleolus_total_signal'
else:
    # fallback: sum of per-nucleolus means (less ideal; consider switching to total)
    agg_signal_col = 'peptide_nucleol_intensity'

cell_totals = (
    feature_information
    .groupby(['image_name', 'nuc_number'])
    .agg(
        nucleolar_total=(agg_signal_col, 'sum'),
        nuc_intensity_mean=('nuc_intensity_mean', 'first'),
        nuc_size=('nuc_size', 'first')
    )
    .reset_index()
)

# 2) total nuclear peptide signal (mean × area in pixels)
cell_totals['nucleus_total_signal'] = cell_totals['nuc_intensity_mean'] * cell_totals['nuc_size']

# 3) enrichment fraction
cell_totals['nucleolar_enrichment'] = np.where(
    cell_totals['nucleus_total_signal'] > 0,
    cell_totals['nucleolar_total'] / cell_totals['nucleus_total_signal'],
    np.nan
)

# Merge enrichment back to every nucleolus row of the same cell
feature_information = feature_information.merge(
    cell_totals[['image_name', 'nuc_number', 'nucleolar_enrichment']],
    on=['image_name', 'nuc_number'],
    how='left'
)
# =================== END: Correct cell-level enrichment ======================

# save data for plotting coords
feature_information.to_csv(f'{output_folder}R-Loops_feature_info.csv', index=False)



# make additional df for avgs per replicate
features_of_interest = ['nucleolar_area', 'nucleolar_eccentricity',
       'nucleolar_major_axis_length', 'nucleolar_minor_axis_length',
       'nucleolar_perimeter', 'peptide_nucleol_cv',
       'peptide_nucleol_skew', 'peptide_nucleol_intensity', 'nuc_size', 'nuc_intensity_mean', 'peptide_nucleol_aspect_ratio',
       'peptide_nucleol_circularity', 'nucleolar_enrichment', 'per_nucleolus_total_signal']

nucleol_summary_reps = []
for col in features_of_interest:
    reps_table = feature_information.groupby(['condition', 'rep']).mean(numeric_only=True)[f'{col}']
    nucleol_summary_reps.append(reps_table)
nucleol_summary_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['condition', 'rep'], how='outer'), nucleol_summary_reps).reset_index()


nucleol_summary_reps_df.to_csv(
    os.path.join(output_folder, 'per-nucleolus_summary_reps.csv'),
    index=False
)



# --------------visualize calculated parameters - raw --------------
x = 'condition'
order = [
    'DFMO','DFMO-1hrSpermidine','DFMO-CPT-1hrSpermidine',
    'DFMO-2hrSpermidine','DFMO-4hrSpermidine','DFMO-24hrSpermidine'
]

plots_per_fig = 6
num_features = len(features_of_interest)
num_figures = math.ceil(num_features / plots_per_fig)

for fig_num in range(num_figures):
    start_idx = fig_num * plots_per_fig
    end_idx = min(start_idx + plots_per_fig, num_features)
    current_features = features_of_interest[start_idx:end_idx]
    n = len(current_features)

    # grid: 2 rows x up to 3 cols (since plots_per_fig=6)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, parameter in enumerate(current_features):
        ax = axes[i]
        sns.stripplot(
            data=feature_information, x=x, y=parameter,
            order=order, size=8, alpha=0.4, linewidth=1, edgecolor='white',
            dodge=False, ax=ax
        )
        sns.boxplot(
            data=feature_information, x=x, y=parameter,
            order=order, palette=['.9'], ax=ax
        )
        ax.set_title(parameter, fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelrotation=45)
        # >>> center the tick labels (key fix)
        for lab in ax.get_xticklabels():
            lab.set_ha('center')
        sns.despine(ax=ax)

    # hide unused axes (only relevant if last page has <6 panels)
    for j in range(n, nrows * ncols):
        fig.delaxes(axes[j])

    # add centered title after layout
    fig.suptitle(f'Calculated Parameters - per nucleol (Fig {fig_num + 1})',
                 fontsize=18, ha='center', y=0.995)
    fig.subplots_adjust(top=0.90)  # leave headroom for the title

    output_path = f'{output_folder}/puncta-features_pernucleol_raw_fig{fig_num + 1}.png'
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close(fig)

# --- Plotting specific plots --------------
x = 'condition'
order = [
    'DFMO','DFMO-1hrSpermidine','DFMO-CPT-1hrSpermidine',
    'DFMO-2hrSpermidine','DFMO-4hrSpermidine','DFMO-24hrSpermidine'
]
features_of_interest = ['peptide_nucleol_intensity', 'nuc_intensity_mean', 'nucleolar_enrichment']

plots_per_fig = 1  # one feature per figure (three separate files)
num_features = len(features_of_interest)
num_figures = math.ceil(num_features / plots_per_fig)

# ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

for fig_num in range(num_figures):
    plt.figure(figsize=(20, 8))
    plt.subplots_adjust(hspace=0.5)

    start_idx = fig_num * plots_per_fig
    end_idx = min(start_idx + plots_per_fig, num_features)
    current_features = features_of_interest[start_idx:end_idx]

    # helpful for title/filename
    feature_label = current_features[0] if len(current_features) == 1 else f"features_{start_idx+1}-{end_idx}"

    plt.suptitle(f'per Nucleol - fig {fig_num + 1}', fontsize=18, y=0.99)

    for i, parameter in enumerate(current_features):
        ax = plt.subplot(2, 3, i + 1)
        sns.stripplot(
            data=feature_information, x=x, y=parameter,
            dodge=True, edgecolor='white', linewidth=1, size=8, alpha=0.4,
            order=order, ax=ax
        )
        sns.boxplot(
            data=feature_information, x=x, y=parameter,
            palette=['.9'], order=order, ax=ax
        )
        ax.set_title(parameter, fontsize=12)
        ax.set_xlabel('')
        plt.xticks(rotation=45)
        sns.despine()

    plt.tight_layout()

    # unique filename per figure:
    # e.g., perNucleol_fig1_peptide_nucleol_intensity.png
    output_path = os.path.join(
        output_folder,
        f'perNucleol_fig{fig_num + 1}_{feature_label}.png'
    )
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()  # close to avoid overwriting/bleed between figs

#------Plotting average of each rep --------------
# --- Plotting: all points, rep means (black), boxplot (hide outliers), and n per condition ---
x = 'condition'
order = [
    'DFMO','DFMO-1hrSpermidine','DFMO-CPT-1hrSpermidine',
    'DFMO-2hrSpermidine','DFMO-4hrSpermidine','DFMO-24hrSpermidine'
]

features_of_interest = ['peptide_nucleol_intensity']
quantile_crop = (0.01, 0.99)

os.makedirs(plotting_folder, exist_ok=True)

for parameter in features_of_interest:
    # a bit wider + taller so labels have room
    fig, ax = plt.subplots(figsize=(14, 12))

    # scatter, box, replicate means
    sns.stripplot(data=feature_information, x=x, y=parameter,
                  order=order, size=3.0, alpha=0.25, linewidth=0,
                  dodge=False, ax=ax)
    sns.boxplot(data=feature_information, x=x, y=parameter,
                order=order, color='white', width=0.6,
                showfliers=False, ax=ax)
    sns.stripplot(data=nucleol_summary_reps_df, x=x, y=parameter,
                  order=order, color='k', size=8, linewidth=0,
                  dodge=False, jitter=False, ax=ax, zorder=5)

    # ---- Format x labels nicely (2-line condition + 1 line for n) ----
    n_counts = feature_information.groupby(x).size().reindex(order).fillna(0).astype(int)
    new_labels = []
    for cond in order:
        main = cond.replace('-', '\n', 1)  # split into two lines at the first hyphen
        new_labels.append(f"{main}\n(n={int(n_counts.loc[cond])})")

    # apply with rotation + right anchoring to avoid overlap
    ax.set_xticklabels(
        new_labels,
        rotation=30,                      # tilt for more horizontal space
        ha='right',                       # anchor to the right edge
        rotation_mode='anchor',
        fontsize=11,                      # slightly smaller labels
    )

    # Optional y-crop (unchanged)
    if quantile_crop is not None:
        lo_q, hi_q = quantile_crop
        series = feature_information[parameter].dropna()
        if len(series) > 0:
            y_low = series.quantile(lo_q)
            y_high = series.quantile(hi_q)
            if np.isfinite(y_low) and np.isfinite(y_high) and y_high > y_low:
                ax.set_ylim(y_low, y_high)

    # Titles / axes
    ax.set_title('Nucleolus Intensities per Nucleoli', fontsize=20, pad=16)
    ax.set_xlabel('')
    ax.set_ylabel('Intensity (AU)', fontsize=15)
    ax.tick_params(axis='x', length=0)    # hide tick marks; just keep labels
    sns.despine()

    # Give labels room at the bottom; leave headroom for the title
    fig.subplots_adjust(bottom=0.35, top=0.90)

    out_png = os.path.join(plotting_folder, f'perNucleol_{parameter}.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)



## Below will measure and plot per cell instead of per nucleolus
# --------------Grab major and minor_axis_length for punctas--------------
minor_axis = feature_information.groupby(
    ['image_name', 'nuc_number'])['nucleolar_minor_axis_length'].mean()
major_axis = feature_information.groupby(
    ['image_name', 'nuc_number'])['nucleolar_major_axis_length'].mean()

# --------------Calculate average size of punctas per nuc--------------
puncta_avg_area = feature_information.groupby(
    ['image_name', 'nuc_number'])['nucleolar_area'].mean().reset_index()

# --------------Calculate proportion of area in punctas--------------
nuc_size = feature_information.groupby(
    ['image_name', 'nuc_number'])['nuc_size'].mean()
puncta_area = feature_information.groupby(
    ['image_name', 'nuc_number'])['nucleolar_area'].sum()
puncta_proportion = ((puncta_area / nuc_size) *
                   100).reset_index().rename(columns={0: 'proportion_puncta_area'})

# --------------Calculate number of 'punctas' per nuc--------------
puncta_count = feature_information.groupby(
    ['image_name', 'nuc_number'])['nucleolar_area'].count()

# --------------Calculate average size of punctas per nuc--------------
avg_eccentricity = feature_information.groupby(
    ['image_name', 'nuc_number'])['nucleolar_eccentricity'].mean().reset_index()

# --------------Grab nuc nucleol cov --------------
nucleol_cv_mean = feature_information.groupby(
    ['image_name', 'nuc_number'])['peptide_nucleol_cv'].mean()

# --------------Grab nuc nucleol skew --------------
nucleol_skew_mean = feature_information.groupby(
    ['image_name', 'nuc_number'])['peptide_nucleol_skew'].mean()

# --------------Grab nucleolar enrichment --------------
nucleolar_enrichment = feature_information.groupby(
    ['image_name', 'nuc_number'])['nucleolar_enrichment'].mean()

# --------------Grab nuc intensity mean --------------
nuc_intensity_mean = feature_information.groupby(
    ['image_name', 'nuc_number'])['nuc_intensity_mean'].mean()

# ---- Grab the nucleolar intensity per cell --------
per_nucleolus_total_signal = feature_information.groupby(
    ['image_name', 'nuc_number'])['per_nucleolus_total_signal'].mean()

# --------------Summarise, save to csv--------------
summary = functools.reduce(lambda left, right: pd.merge(left, right, on=['image_name', 'nuc_number'], how='outer'), [nuc_size.reset_index(), puncta_avg_area, puncta_proportion, puncta_count.reset_index(), minor_axis, major_axis, avg_eccentricity, nucleol_cv_mean, nucleol_skew_mean, nucleolar_enrichment, nuc_intensity_mean,per_nucleolus_total_signal])
summary.columns = ['image_name', 'nuc_number',  'nuc_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'nucleol_cv_mean', 'nucleol_skew_mean', 'nucleolar_enrichment', 'nuc_intensity_mean','per_nucleolus_total_signal']

# --------------tidy up dataframe--------------
# add columns for sorting
# add peptide name
summary['condition'] = summary['image_name'].str.split('_').str[1]
summary['rep'] = summary['image_name'].str.extract(r'_(\d+)-\d+$')



# save
summary.to_csv(f'{output_folder}puncta_detection_summary.csv')

features_of_interest = [ 'nuc_size', 'mean_puncta_area',
       'puncta_area_proportion', 'puncta_count', 'puncta_mean_minor_axis',
       'puncta_mean_major_axis', 'avg_eccentricity', 'nucleol_cv_mean',
       'nucleol_skew_mean', 'nucleolar_enrichment', 'nuc_intensity_mean','per_nucleolus_total_signal']

# --------------visualize calculated parameters - raw --------------

x = 'condition'
order = [
    'DFMO','DFMO-1hrSpermidine','DFMO-CPT-1hrSpermidine',
    'DFMO-2hrSpermidine','DFMO-4hrSpermidine','DFMO-24hrSpermidine'
]
plots_per_fig = 6
num_features = len(features_of_interest)
num_figures = math.ceil(num_features / plots_per_fig)

for fig_num in range(num_figures):
    # Create a new figure
    plt.figure(figsize=(20, 8))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'Calculated Parameters - per cell (Fig {fig_num + 1})', fontsize=18, y=0.99)

    # Get the current slice of features
    start_idx = fig_num * plots_per_fig
    end_idx = min(start_idx + plots_per_fig, num_features)
    current_features = features_of_interest[start_idx:end_idx]

    for i, parameter in enumerate(current_features):
        ax = plt.subplot(2, 3, i + 1)
        sns.stripplot(data=summary, x=x, y=parameter, dodge=True, edgecolor='white', linewidth=1, size=8, alpha=0.4, order=order, ax=ax)
        sns.boxplot(data=summary, x=x, y=parameter, palette=['.9'], order=order, ax=ax)
        ax.set_title(parameter, fontsize=12)
        ax.set_xlabel('')
        plt.xticks(rotation=45)
        sns.despine()

    plt.tight_layout()

    output_path = f'{output_folder}/features_percell_raw_fig{fig_num + 1}.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()




# --- Plotting specific plots --------------
# your settings
x = 'condition'
order = [
    'DFMO','DFMO-1hrSpermidine','DFMO-CPT-1hrSpermidine',
    'DFMO-2hrSpermidine','DFMO-4hrSpermidine','DFMO-24hrSpermidine'
]
features_of_interest = ['nuc_intensity_mean', 'nucleolar_enrichment']

# compute N for each x-group (aligned to 'order')
group_counts = summary[x].value_counts().reindex(order, fill_value=0)
x_labels_with_n = [f"{cond}\n(n={group_counts[cond]})" for cond in order]

plots_per_fig = 1
num_features = len(features_of_interest)
num_figures = math.ceil(num_features / plots_per_fig)

os.makedirs(output_folder, exist_ok=True)

for fig_num in range(num_figures):
    fig = plt.figure(figsize=(20, 8))

    start_idx = fig_num * plots_per_fig
    end_idx = min(start_idx + plots_per_fig, num_features)
    current_features = features_of_interest[start_idx:end_idx]

    feature_label = current_features[0] if len(current_features) == 1 else f"features_{start_idx+1}-{end_idx}"

    for i, parameter in enumerate(current_features):
        ax = plt.subplot(2, 3, i + 1)

        sns.stripplot(
            data=summary, x=x, y=parameter,
            dodge=True, edgecolor='white', linewidth=1, size=8, alpha=0.4,
            order=order, ax=ax
        )
        sns.boxplot(
            data=summary, x=x, y=parameter,
            palette=['.9'], order=order, ax=ax
        )
        ax.set_title(parameter, fontsize=12)
        ax.set_xlabel('')
        ax.set_xticklabels(x_labels_with_n, rotation=45, ha='right')
        sns.despine()

    # adjust layout first, reserving space at the top
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # now add centered title
    fig.suptitle(f'per cell - fig {fig_num + 1}', fontsize=18, ha='center')

    output_path = os.path.join(output_folder, f'per-cell_fig{fig_num + 1}_{feature_label}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()



#------Plotting average of each rep --------------

cell_summary_reps = []
for col in features_of_interest:
    reps_series = summary.groupby(['condition', 'rep']).mean(numeric_only=True)[col]
    reps_df = reps_series.to_frame(name=col).reset_index()  # columns: condition, rep, <col>
    cell_summary_reps.append(reps_df)

cell_summary_reps_df = functools.reduce(
    lambda left, right: pd.merge(left, right, on=['condition', 'rep'], how='outer'),
    cell_summary_reps
)

cell_summary_reps_df.to_csv(
    os.path.join(output_folder, 'cell_summary_reps.csv'),
    index=False
)

# --- Plotting: all points, rep means (black), boxplot (hide outliers), and n per condition ---

# --- Custom axis labels ---
y_labels = {
    'nuc_intensity_mean': 'Mean Nuclear Intensity (A.U.)',
    'nucleolar_enrichment': 'Nucleolar Enrichment Ratio',
    'peptide_nucleol_intensity': 'Nucleolus Intensity (A.U.)',
}

x_labels = {
    'DFMO': 'DFMO only',
    'DFMO-1hrSpermidine': '1 hr Spermidine',
    'DFMO-CPT-1hrSpermidine': '1 hr Spermidine + CPT',
    'DFMO-2hrSpermidine': '2 hr Spermidine',
    'DFMO-4hrSpermidine': '4 hr Spermidine',
    'DFMO-24hrSpermidine': '24 hr Spermidine'
}

# --- Plot loop ---
for parameter in features_of_interest:
    fig, ax = plt.subplots(figsize=(12, 7))

    # === STYLE SETUP ===
    sns.set_theme(style="white", context="talk", font_scale=1.2)
    palette = sns.color_palette("muted", n_colors=len(order))

    # === STRIP (individual cells/nucleoli) ===
    sns.stripplot(
        data=summary, x=x, y=parameter,
        order=order, palette=palette,
        size=2.8, alpha=0.35, linewidth=0,
        jitter=0.2, ax=ax, zorder=1
    )

    # === BOX ===
    sns.boxplot(
        data=summary, x=x, y=parameter,
        order=order, width=0.6,
        showcaps=True, showfliers=False,
        boxprops={'facecolor': '0.96', 'edgecolor': '0.3', 'linewidth': 1.1},
        medianprops={'color': '0.25', 'linewidth': 1.4},
        whiskerprops={'color': '0.3', 'linewidth': 1.1},
        capprops={'color': '0.3', 'linewidth': 1.1},
        ax=ax, zorder=2
    )

    # === REPLICATE MEANS (black with white outline) ===
    sns.stripplot(
        data=cell_summary_reps_df, x=x, y=parameter,
        order=order, color='black', size=7,
        edgecolor='white', linewidth=1.2,
        jitter=False, ax=ax, zorder=5
    )

    # === CLEANUP & FORMATTING ===
    sns.despine(offset=8)
    ax.set_xlabel('')
    ax.set_ylabel(
        y_labels.get(parameter, parameter.replace('_', ' ').title()),
        fontsize=16
    )
    ax.set_title(
        parameter.replace('_', ' ').title(),
        fontsize=20, pad=22, weight='bold'
    )

    # === MULTI-LINE X LABELS WITH N ===
    n_counts = summary.groupby(x).size().reindex(order).fillna(0).astype(int)
    formatted_labels = []
    for cond in order:
        short_label = x_labels.get(cond, cond)  # lookup or fallback
        formatted_labels.append(f"{short_label}\n(n={n_counts.loc[cond]})")

    # ✅ apply labels AFTER loop (not inside it)
    ax.set_xticklabels(
        formatted_labels,
        ha='right',
        rotation=30,
        rotation_mode='anchor',
        fontsize=13
    )

    # === OPTIONAL Y-CROP ===
    if quantile_crop is not None:
        lo_q, hi_q = quantile_crop
        vals = summary[parameter].dropna()
        if len(vals) > 0:
            y_low, y_high = vals.quantile(lo_q), vals.quantile(hi_q)
            if np.isfinite(y_low) and np.isfinite(y_high) and y_high > y_low:
                ax.set_ylim(y_low, y_high)

    # === POLISH ===
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', which='major', width=1.2, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('0.3')

    plt.subplots_adjust(bottom=0.4, top=0.9)  # extra room for rotated x labels

    # === SAVE ===
    out_png = os.path.join(plotting_folder, f'per-cell_{parameter}.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

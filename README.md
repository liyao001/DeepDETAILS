# DeepDETAILS: Deep-learning-based DEconvolution of Tissue profiles with Accurate Interpretation of Locus-specific Signals

---

![Supported platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx-lightgrey.svg)
![Supported Python versions](https://img.shields.io/badge/python-3.x-blue.svg)
[![PyPI](https://github.com/liyao001/DeepDETAILS/actions/workflows/publish.yml/badge.svg)](https://github.com/liyao001/DeepDETAILS/actions/workflows/publish.yml)
[![DeepDETAILS compendium](https://img.shields.io/website?label=DeepDETAILS%20compendium&url=https%3A%2F%2Fdetails.yulab.org)](//details.yulab.org)


## Installation

DeepDETAILS can be installed via `pip`: 

```console
pip install DeepDETAILS
```

To export the results to bigWig files, please make sure you have [`bedGraphToBigWig`](https://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bedGraphToBigWig) 
and [`bedtools`](https://bedtools.readthedocs.io/en/latest/) installed.

## Get started

### Step 1: Prepare datasets for deconvolution
DeepDETAILS requires the following input files:
* Strand-specific signals for the bulk library (bigWig format)
* Region of interests (e.g. peaks) in the bulk library (bed format)
* Aligned fragments from the reference sc/snATAC-seq (bed-like tabular format, required columns: chrom, chromStart, chromEnd, barcode, and readSupport). Example
* Cell type annotation for each cell barcode (tabular format, required columns: barcode and cell type annotation).
* Reference genome sequence (fasta format).
* Chromosome size.

```shell
deepdetails prep-data \
    --bulk-pl bulk.pl.bw \
    --bulk-mn bulk.mn.bw \
    --regions bulk.peaks.bed \
    --fragments fragments.tsv.gz \
    --barcodes barcodes.tsv \
    --accessible-regions atac_peaks.bed \
    --save-to ./dataset \
    --genome-fa GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom-size chrNameLength.txt
```

### Step 2: Deconvolution
After building the dataset folder, you can run the deconvolution process (requires GPU):
```shell
deepdetails deconv \
    --dataset ./dataset \
    --save-to . \
    --study-name sample-a
```
The outputs from a successful deconvolution process look like the following:
```
.
├── sample-a
│   └── 250212144109: The folder containing deconvolution results (name changes according to the time).
│       ├── epoch=0-step=2538.ckpt: Trained model
│       ├── hparams.yaml: Hyperparamters
│       ├── metrics.csv: Training log
...
│       └── preview972.0.131072.0000.s21250212144109.png: Preview genome browser views
├── sample-a.counts.csv.gz: Deconvolved read counts (1-kb resolution) for each cell type
└── sample-a.predictions.h5: Deconvolved signal (1-bp resolution) for each cell type
```

### Step 3: Visualize the results (optional)
This step exports deconvolved signal tracks (bigWig) to visualize the signals in each cell type, and it's optional. 
You need to locate the exported predictions from the previous step by looking for files like `sample-name.predictions.h5`. 
After you get the file, you can run the following command:
```shell
deepdetails build-bw \
    -p sample-name.predictions.h5 \
    --save-to . \
    --chrom-size chrNameLength.txt
```
You should be able to see deconvolved signal tracks for each cell type 
(named like `cell_type.pl.bw` / `cell_type.mn.bw`) 
in the output directory after the command finishes.


## Reference
> Yao, L. et al. High-resolution reconstruction of cell-type specific transcriptional regulatory processes from bulk sequencing samples. [Preprint at bioRxiv](https://doi.org/10.1101/2025.04.02.646189) (2025).

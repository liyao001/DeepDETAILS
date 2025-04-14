import unittest
import gzip
import os
import h5py
import pandas as pd
import pyfaidx
import pyBigWig
import numpy as np
from deepdetails.protocols import prepare_dataset


def _decompress_geneome_fa(input_file: str):
    output_file = input_file.replace(".gz", "")
    with gzip.open(input_file, "rb") as f_in:
        with open(output_file, "wb") as f_out:
            f_out.write(f_in.read())
    return output_file


class DatasetTestCase(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.fasta = os.path.join(current_dir, "data/chr22.fa")
        self.fasta_obj = pyfaidx.Fasta(self.fasta)
        self.chrom_size = os.path.join(current_dir, "data/hg38.chrom.sizes")
        self.bulk_pl = os.path.join(current_dir, "data/bulk.chr22.pl.bw")
        self.bulk_mn = os.path.join(current_dir, "data/bulk.chr22.mn.bw")
        self.bulk_objs = [pyBigWig.open(f) for f in (self.bulk_pl, self.bulk_mn)]
        self.acc_files = (
            os.path.join(current_dir, "data/acc.K562.chr22.bw"),
            os.path.join(current_dir, "data/acc.GM12878.chr22.bw"),
            os.path.join(current_dir, "data/acc.MCF7.chr22.bw")
        )

        self.pl_ref_files = (
            os.path.join(current_dir, "data/K562.ds.chr22.pl.bw"),
            os.path.join(current_dir, "data/GM12878.ds.chr22.pl.bw"),
            os.path.join(current_dir, "data/MCF7.ds.chr22.pl.bw")
        )
        self.mn_ref_files = (
            os.path.join(current_dir, "data/K562.ds.chr22.mn.bw"),
            os.path.join(current_dir, "data/GM12878.ds.chr22.mn.bw"),
            os.path.join(current_dir, "data/MCF7.ds.chr22.mn.bw")
        )
        self.ref_objs = [(pyBigWig.open(pf), pyBigWig.open(mf)) for pf, mf in zip(self.pl_ref_files, self.mn_ref_files)]

        self.ref_labels = ("K562", "GM12878", "MCF7")

        self.acc_objs = [pyBigWig.open(f) for f in self.acc_files]
        self.t_x = 4096
        self.sampling_range_start = 21700000 + 2 * self.t_x  # start of q11.22
        self.sampling_range_end = 49100000 - 2 * self.t_x  # end of q13.32

    @classmethod
    def setUpClass(cls):
        super(DatasetTestCase, cls).setUpClass()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fasta = os.path.join(current_dir, "data/chr22.fa")
        if os.path.exists(fasta + ".gz") and not os.path.exists(fasta):
            _decompress_geneome_fa(fasta + ".gz")

    def test_typical_deconv_dataset(self):
        """
        In this test case, we try to test when the user gets a bulk library as well as cell type specific
        accessibility information, and we try to build a dataset.

        Here we are goning to test the DNA sequences, bulk and cell-type specific accessibility signals
        """
        n_samples = 16
        starts = np.random.randint(low=self.sampling_range_start, high=self.sampling_range_end, size=n_samples)
        ends = starts + self.t_x
        regions = pd.DataFrame({0: "chr22", 1: starts, 2: ends, 3: np.ones_like(starts)})
        _tmp_peak_file = "regions.bed"
        regions.to_csv(_tmp_peak_file, sep="\t", header=False, index=False)

        # build dataset
        prepare_dataset(
            regions=(_tmp_peak_file, ),
            bulk_pl=self.bulk_pl,
            bulk_mn=self.bulk_mn,
            accessibility=list(self.acc_files),
            window_size=self.t_x,
            chrom_size=self.chrom_size,
            genome_fa=self.fasta.replace(".gz", ""),
            save_to="."
        )
        # the above call should generate data.h5 and regions.csv files
        self.assertTrue(os.path.exists("data.h5"))
        self.assertTrue(os.path.exists("regions.csv"))

        compiled_regions = pd.read_csv("regions.csv", header=None)
        h5ds = h5py.File("data.h5", "r")
        sample_idx = np.random.randint(low=0, high=n_samples-1, size=1)[0]
        coord = compiled_regions.loc[sample_idx]
        ref_seq = self.fasta_obj[coord[0]][coord[1]:coord[2]].seq.upper()
        seq_in_ds = "".join(["ACGT"[i] for i in h5ds["dec"]["seq"][sample_idx, :, :].argmax(axis=0)]).upper()

        # check sequence
        self.assertTrue(seq_in_ds == ref_seq or ref_seq.find("N") != -1)

        # check bulk signal
        self.assertTrue(np.all(h5ds["dec/bulk"][sample_idx, 0, :] == np.abs(
            np.nan_to_num(self.bulk_objs[0].values(coord[0], coord[1], coord[2], numpy=True)))))
        self.assertTrue(np.all(h5ds["dec/bulk"][sample_idx, 1, :] == np.abs(
            np.nan_to_num(self.bulk_objs[1].values(coord[0], coord[1], coord[2], numpy=True)))))

        # check acc signal
        for offset, ref_acc in enumerate(self.acc_objs):
            self.assertTrue(np.all(h5ds["dec/acc"][sample_idx, offset, :] == np.abs(
                np.nan_to_num(ref_acc.values(coord[0], coord[1], coord[2], numpy=True)))))

    def test_simulated(self):
        """
        In this test case, we try to test when the user gets a bulk library, cell type specific
        accessibility information, as well as the ground truth signal, and we try to build a dataset.
        """
        n_samples = 16
        starts = np.random.randint(low=self.sampling_range_start, high=self.sampling_range_end, size=n_samples)
        ends = starts + self.t_x
        regions = pd.DataFrame({0: "chr22", 1: starts, 2: ends, 3: np.ones_like(starts)})
        _tmp_peak_file = "regions.bed"
        regions.to_csv(_tmp_peak_file, sep="\t", header=False, index=False)
        workdir = "data_with_ref"

        if not os.path.exists(workdir):
            os.mkdir(workdir)

        # build dataset
        prepare_dataset(
            regions=(_tmp_peak_file,),
            bulk_pl=self.bulk_pl,
            bulk_mn=self.bulk_mn,
            accessibility=list(self.acc_files),
            window_size=self.t_x,
            chrom_size=self.chrom_size,
            genome_fa=self.fasta.replace(".gz", ""),
            save_to=f"./{workdir}", ref_labels=self.ref_labels,
            ref_pls=self.pl_ref_files, ref_mns=self.mn_ref_files
        )
        # the above call should generate data.h5 and regions.csv files
        self.assertTrue(os.path.exists(f"{workdir}/data.h5"))
        self.assertTrue(os.path.exists(f"{workdir}/regions.csv"))

        compiled_regions = pd.read_csv(f"{workdir}/regions.csv", header=None)
        h5ds = h5py.File(f"{workdir}/data.h5", "r")
        sample_idx = np.random.randint(low=0, high=n_samples - 1, size=1)[0]
        coord = compiled_regions.loc[sample_idx]
        ref_seq = self.fasta_obj[coord[0]][coord[1]:coord[2]].seq.upper()
        seq_in_ds = "".join(["ACGT"[i] for i in h5ds["dec"]["seq"][sample_idx, :, :].argmax(axis=0)]).upper()

        # check reference
        # - check key exists
        self.assertTrue("ref" in h5ds["dec"])
        # - check signal values
        for i, ref_pairs in enumerate(self.ref_objs):
            self.assertTrue(np.all(h5ds["dec/ref"][sample_idx, 2 * i, :] == np.abs(
                np.nan_to_num(ref_pairs[0].values(coord[0], coord[1], coord[2], numpy=True)))))
            self.assertTrue(np.all(h5ds["dec/ref"][sample_idx, 2 * i + 1, :] == np.abs(
                np.nan_to_num(ref_pairs[1].values(coord[0], coord[1], coord[2], numpy=True)))))

        # check sequence
        self.assertTrue(seq_in_ds == ref_seq or ref_seq.find("N") != -1)

        # check bulk signal
        self.assertTrue(np.all(h5ds["dec/bulk"][sample_idx, 0, :] == np.abs(
            np.nan_to_num(self.bulk_objs[0].values(coord[0], coord[1], coord[2], numpy=True)))))
        self.assertTrue(np.all(h5ds["dec/bulk"][sample_idx, 1, :] == np.abs(
            np.nan_to_num(self.bulk_objs[1].values(coord[0], coord[1], coord[2], numpy=True)))))

        # check acc signal
        for offset, ref_acc in enumerate(self.acc_objs):
            self.assertTrue(np.all(h5ds["dec/acc"][sample_idx, offset, :] == np.abs(
                np.nan_to_num(ref_acc.values(coord[0], coord[1], coord[2], numpy=True)))))


if __name__ == '__main__':
    unittest.main()

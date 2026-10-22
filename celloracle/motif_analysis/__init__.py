# -*- coding: utf-8 -*-
"""
The :mod:`.motif_analysis` module implements transcription factor motif scan.

Genomic activity information (peak of ATAC-seq or ChIP-seq) is extracted first.
Then the peak DNA sequence will be subjected to TF motif scan.
Finally we will get list of TFs that potentially binds to a specific gene.

This sub-package depends on the *optional* ``genomepy`` and ``gimmemotifs``
libraries.  They are not needed for the core GRN-inference and simulation
workflow, so importing :mod:`celloracle` works without them.  Accessing any
object here without the optional dependencies installed raises a descriptive
:class:`ImportError`.
"""

_import_error = None

try:
    from .motif_analysis_utility import is_genome_installed
    from .process_bed_file import peak2fasta, read_bed, remove_zero_seq, check_peak_format
    from .tfinfo_core import (load_TFinfo, load_TFinfo_from_parquets,
                              make_TFinfo_from_scanned_file, TFinfo, scan_dna_for_motifs)
    from .reference_genomes import SUPPORTED_REF_GENOME
    from .tss_annotation import get_tss_info
    from .process_cicero_data import integrate_tss_peak_with_cicero
    from . import process_cicero_data
    from .motif_data import load_motifs, MOTIFS_LIST

    __all__ = ["is_genome_installed", "peak2fasta", "read_bed", "remove_zero_seq", "check_peak_format",
               "load_TFinfo", "load_TFinfo_from_parquets",
               "make_TFinfo_from_scanned_file",
               "TFinfo", "scan_dna_for_motifs", "SUPPORTED_REF_GENOME",
               "get_tss_info", "process_cicero_data",
               "integrate_tss_peak_with_cicero",
               "load_motifs", "MOTIFS_LIST"]
except ImportError as exc:  # optional dependencies not installed
    _import_error = exc
    __all__ = []

    _MISSING_MESSAGE = (
        "CellOracle's motif_analysis module requires the optional dependencies "
        "'genomepy' and 'gimmemotifs'.\n"
        "Install them with:\n\n"
        "    pip install celloracle[motif]\n\n"
        "or\n\n"
        "    pip install genomepy gimmemotifs\n"
    )

    def __getattr__(name):
        raise ImportError(_MISSING_MESSAGE) from _import_error

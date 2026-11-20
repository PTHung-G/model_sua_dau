"""ViDiCRF v2: Phonology-aware Conformer-RoPE + Anchor-Propagating Trigram CRF."""

from .orthography import (
    strip_diac,
    has_diacritic,
    normalize_tone_placement,
    is_valid_vn_or_clean_candidate,
    parse_syllable_phonology,
    compact_inventory,
)
try:  # optional module; not needed by the v2/v3 core pipeline
    from .v1_mbr_decoder import (
        forward_backward_marginals,
        mbr_decode,
        constrained_viterbi,
    )
except ImportError:  # pragma: no cover
    pass
from .model import (
    ViDiCRFv2,
    trigram_log_partition,
    trigram_gold_score,
    trigram_viterbi,
    trigram_marginals,
    compute_loss,
)
from .data_pipeline import (
    V2Inventory,
    EncodedSplitV2,
    build_v2_inventory,
    encode_corpus_v2,
    make_batch_v2,
)
from .restorer import FastRestorerV2

__all__ = [
    "strip_diac",
    "has_diacritic",
    "normalize_tone_placement",
    "is_valid_vn_or_clean_candidate",
    "parse_syllable_phonology",
    "compact_inventory",
    "forward_backward_marginals",
    "mbr_decode",
    "constrained_viterbi",
    "ViDiCRFv2",
    "trigram_log_partition",
    "trigram_gold_score",
    "trigram_viterbi",
    "trigram_marginals",
    "compute_loss",
    "V2Inventory",
    "EncodedSplitV2",
    "build_v2_inventory",
    "encode_corpus_v2",
    "make_batch_v2",
    "FastRestorerV2",
]

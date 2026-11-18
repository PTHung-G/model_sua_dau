"""Tests: punctuation preservation, gap alignment, inventory isolation, model shape."""
import json
import re
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'charexp'))

import build_punct_tensors_v1 as B


def test_tokenizer_keeps_alnum_runs_only():
    source = 'Hôm nay tôi buồn, bạn có yêu tôi không?'
    toks = [m.group() for m in B.TOK.finditer(source)]
    assert toks == ['Hôm', 'nay', 'tôi', 'buồn', 'bạn', 'có', 'yêu', 'tôi', 'không']
    assert 'buồn' in toks and 'bạn' in toks


def test_gaps_capture_every_punctuation_string():
    source = '“Đừng đi!” — cô ấy nói; rồi hỏi: (vì sao?)'
    ms, gs = B.gaps(source)
    assert len(ms) == len(gs)
    joined = ''.join(pre for pre, _ in gs) + gs[-1][1]
    for ch in '“”!—;:(?)':
        assert ch in joined, f'punctuation {ch!r} lost'
    # Adjacent words inside quotes must expose the quote+space gap, not a blank one.
    assert gs[0][0] == '“'
    assert '!' in gs[1][1]


def test_gaps_preserve_double_space_and_decimal():
    ms, gs = B.gaps('Giá 3,14 đồng...  Đúng chưa')
    assert len(ms) == 6
    assert '  ' in gs[3][1]
    assert ',' in gs[1][1]


def test_build_and_encode_alignment(tmp_path):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    rows = [
        {'input': 'Hom nay toi buon, ban co yeu toi khong?', 'target': 'Hôm nay tôi buồn, bạn có yêu tôi không?'},
        {'input': 'Anh ay noi: "Toi di lam."', 'target': 'Anh ấy nói: "Tôi đi làm."'},
    ]
    for name in ('train', 'validation', 'test'):
        with (corpus / (name + '.jsonl')).open('w', encoding='utf8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    bases, vocab, ctab, pid, n = B.build_table(corpus / 'train.jsonl')
    stats = B.encode(corpus / 'train.jsonl', tmp_path / 'out/train', bases, vocab, ctab, pid)
    assert n == 2 and stats['rows'] == 2 and stats['skipped'] == 0
    sent = np.load(tmp_path / 'out/train.npz')['sent']
    base = np.load(tmp_path / 'out/train.npz')['base']
    gaps = np.load(tmp_path / 'out/train_gaps.npy')
    feats = np.load(tmp_path / 'out/train_input_features.npy')
    assert int(sent.sum()) == len(base)
    assert gaps.shape == (len(base) + len(sent),), 'one gap per word plus trailing gap'
    assert feats.shape == (len(base),)
    # Every word position must resolve to a real candidate row.
    for word_index, base_id in enumerate(base):
        assert ctab[base_id, 0] >= 0
    # Gap stream must contain the comma gap for the first sentence.
    assert len(pid) > 1


def test_inventory_excludes_unseen_gold(tmp_path):
    """A gold form never seen in train must not be silently added by encode."""
    train = tmp_path / 'train.jsonl'
    with train.open('w', encoding='utf8') as f:
        f.write(json.dumps({'input': 'hoa binh', 'target': 'hòa bình'}, ensure_ascii=False) + '\n')
    bases, vocab, ctab, pid, _ = B.build_table(train)
    other = tmp_path / 'other.jsonl'
    with other.open('w', encoding='utf8') as f:
        f.write(json.dumps({'input': 'hoang hon', 'target': 'hoàng hôn'}, ensure_ascii=False) + '\n')
    stats = B.encode(other, tmp_path / 'o/other', bases, vocab, ctab, pid)
    assert stats['rows'] == 0 and stats['skipped'] == 1


def test_punctuation_model_roundtrip():
    import torch
    import sylmodel_punct as M
    nchar, nbase, K = 40, 5, 3
    base_ch = torch.randint(1, nchar, (nbase, 16))
    cand_ch = torch.randint(1, nchar, (K, 16))
    cid = torch.zeros((nbase, K), dtype=torch.int32)
    cid[:, 0] = 0
    cid[1:, 1] = 1
    cid[2:, 2] = 2
    model = M.PunctSylCRF(nchar, base_ch, cand_ch, cid, 10, d=16, hidden=8, r=4, layers=1)
    x = torch.tensor([[1, 2, 3, 1], [2, 3, 0, 0]])
    ln = torch.tensor([4, 2])
    feats = torch.ones((2, 4), dtype=torch.long)
    gaps = torch.tensor([[1, 2, 1, 3, 4], [2, 1, 5, 0, 0]])
    emit, trans, mask = model(x, ln, feats, gaps)
    assert emit.shape == (2, 4, K) and trans.shape == (2, 3, K, K)
    assert mask[1, 2:].sum() == 0, 'padding must be masked out'
    gold = torch.zeros((2, 4), dtype=torch.long)
    loss = M.nll(emit, trans, mask, gold, ln)
    loss.backward()
    assert torch.isfinite(loss)
    # Zero-initialized projections must receive gradients now that they are used.
    assert model.gap_proj.weight.grad is not None
    pred = M.viterbi(emit, trans, mask)
    assert pred.shape == x.shape


def test_gap_change_changes_score():
    """Punctuation context must demonstrably affect the model, not be decorative."""
    import torch
    import sylmodel_punct as M
    torch.manual_seed(0)
    nchar, nbase, K = 40, 5, 3
    model = M.PunctSylCRF(nchar, torch.randint(1, nchar, (nbase, 16)),
                          torch.randint(1, nchar, (K, 16)),
                          torch.zeros((nbase, K), dtype=torch.int32), 10,
                          d=16, hidden=8, r=4, layers=1)
    # The transition gate is identity-initialised, so it must be activated here.
    with torch.no_grad():
        model.gap_proj.weight.normal_(0, 1)
        model.edge_gate.weight.normal_(0, 1)
    x = torch.tensor([[1, 2, 3, 1]])
    ln = torch.tensor([4])
    feats = torch.ones((1, 4), dtype=torch.long)
    a = model(x, ln, feats, torch.tensor([[1, 1, 1, 1, 1]]))
    b = model(x, ln, feats, torch.tensor([[1, 7, 2, 8, 3]]))
    assert not torch.allclose(a[0], b[0]), 'punctuation gaps must change emissions'
    assert not torch.allclose(a[1], b[1]), 'punctuation gaps must change transitions'


def test_identity_at_initialisation():
    """Untrained model must behave exactly like a gap-blind model."""
    import torch
    import sylmodel_punct as M
    torch.manual_seed(0)
    nchar, nbase, K = 40, 5, 3
    model = M.PunctSylCRF(nchar, torch.randint(1, nchar, (nbase, 16)),
                          torch.randint(1, nchar, (K, 16)),
                          torch.zeros((nbase, K), dtype=torch.int32), 10,
                          d=16, hidden=8, r=4, layers=1)
    model.eval()  # dropout would otherwise hide the property under test
    x = torch.tensor([[1, 2, 3, 1]])
    ln = torch.tensor([4])
    feats = torch.ones((1, 4), dtype=torch.long)
    a = model(x, ln, feats, torch.tensor([[1, 1, 1, 1, 1]]))
    b = model(x, ln, feats, torch.tensor([[1, 7, 2, 8, 3]]))
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])

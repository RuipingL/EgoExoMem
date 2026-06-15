#!/usr/bin/env python3
"""
EgoExo multi-view frame selection — two strategies:

  aks:  EgoExo-AKS
        1. CLIP score ego/exo frames independently
        2. Relevance-based budget: K_e = K * sum(s_e) / (sum(s_e) + sum(s_x))
        3. Per-view AKS selection
        4. Merge by timestamp → VLM

  dpp:  EgoExo-DPP
        1-2. Same as AKS
        3. Per-view k-DPP: L_ij = s_i * (f_i · f_j) * s_j,
           greedy MAP via Cholesky update, O(N*C*k)
        4. Same merge

Usage:
  python eval_egoexomem.py --method aks --model internvl \\
    --datasets egoexo lemma --gpu-id 0 \\
    --out results_rag/egoexo_aks_internvl

  python eval_egoexomem.py --method dpp --model qwen \\
    --datasets egoexo lemma --gpu-id 0 \\
    --out results_rag/egoexo_dpp_qwen
"""

import argparse
import heapq
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ── env / paths ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
for _env in [_HERE / ".env", _HERE.parent / ".env",
             _HERE.parent.parent / "data/EgoExo4D/.env"]:
    if _env.exists():
        for _line in open(_env):
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

ROOT       = _HERE.parent.parent / "data"
EGOEXO_QA  = ROOT / "EgoExo4D/QA/merged/qa_merged_v6.json"
EGOEXO_VID = ROOT / "EgoExo4D/takes"
LEMMA_QA   = ROOT / "LEMMA/qa_verified.json"
LEMMA_VID  = ROOT / "LEMMA"

LOCAL_INTERNVL_ID = str(_HERE.parent.parent / "all_frame_concat/InternVL3_5-8B")
LLAVA_MODEL_ID    = str(_HERE.parent.parent / "all_frame_concat/LLaVA-OneVision-1.5-8B-Instruct")
QWEN3VL_MODEL_ID  = str(_HERE.parent.parent / "all_frame_concat/Qwen3-VL-8B-Instruct")
QWEN_MODEL_ID     = str(Path.home() / ".cache/huggingface/hub/"
                        "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
                        "cc594898137f460bfe9f0759e9844b3ce807cfb5")
CLIP_MODEL_ID     = "openai/clip-vit-large-patch14"
SIGLIP_ID         = str(Path.home() / ".cache/huggingface/hub/"
                        "models--google--siglip-so400m-patch14-384/snapshots/"
                        "9fdffc58afc957d1a03a25b10dba0329ab15c2a3")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TOTAL_FRAMES = 32
T1        = 0.8
T2        = -100
ALL_DEPTH = 5

SYSTEM_PROMPT = (
    "You are answering multiple-choice questions about a video. "
    "Selected keyframes from egocentric and exocentric views are provided, "
    "ordered by timestamp. Each frame is labeled with its view [Ego/Exo] and time. "
    "Answer with a single letter (A, B, C, or D) only. No explanation."
)


# ── video helpers ─────────────────────────────────────────────────────────────

def get_video_path(dataset, take_name, view):
    if dataset == "egoexo":
        base = EGOEXO_VID / take_name / "frame_aligned_videos"
        if view == "ego":
            cands = sorted(base.glob("aria*_214-1.mp4")) if base.exists() else []
            return cands[0] if cands else None
        p = base / "cam01.mp4"
        return p if p.exists() else None
    else:
        base = LEMMA_VID / take_name
        p = base / ("fpv1.mp4" if view == "ego" else "master.mp4")
        return p if p.exists() else None


def lemma_take_name(qa_id):
    parts = qa_id.split("_")
    for i, p in enumerate(parts):
        if p.startswith("Q") and p[1:].isdigit():
            return "-".join(parts[:i])
    return qa_id


def extract_frames_1fps(video_path: Path):
    """Return (frame_indices, PIL_images) at 1 FPS."""
    from decord import VideoReader, cpu
    from PIL import Image
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    except Exception:
        return [], []
    fps = max(1, int(vr.get_avg_fps()))
    n_secs = len(vr) // fps
    raw_indices = [j * fps for j in range(n_secs)]
    if not raw_indices:
        return [], []
    frames_np = vr.get_batch(raw_indices).asnumpy()
    pils = [Image.fromarray(frames_np[k]) for k in range(len(raw_indices))]
    return raw_indices, pils


# ── CLIP ──────────────────────────────────────────────────────────────────────

_clip_model     = None
_clip_processor = None


def load_clip(device):
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return
    from transformers import CLIPModel, CLIPProcessor
    print(f"Loading CLIP on {device} ...", flush=True)
    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    print("CLIP loaded.", flush=True)


@torch.no_grad()
def clip_score_and_features(pil_frames: list, question: str, device: str,
                             batch_size: int = 64):
    """
    Returns (scores, features):
      scores:   [N] list of float  — text-image cosine similarity
      features: [N, C] np.ndarray  — normalized CLIP visual embeddings
    """
    if not pil_frames:
        return [], np.zeros((0, 512), dtype=np.float32)

    inp_txt = _clip_processor(text=question, return_tensors="pt",
                               padding=True, truncation=True).to(device)
    txt_feat = _clip_model.get_text_features(**inp_txt)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)  # [1, C]

    all_feats = []
    for i in range(0, len(pil_frames), batch_size):
        batch = pil_frames[i: i + batch_size]
        inp_img = _clip_processor(images=batch, return_tensors="pt").to(device)
        img_feat = _clip_model.get_image_features(**inp_img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        all_feats.append(img_feat.cpu().float())

    vis_feats = torch.cat(all_feats, dim=0).numpy()                   # [N, C]
    scores    = (vis_feats @ txt_feat.cpu().float().numpy().T).ravel() # [N]
    return scores.tolist(), vis_feats


# ── budget allocation ─────────────────────────────────────────────────────────

def budget_split(ego_scores: list, exo_scores: list, K: int):
    """
    K_e = K * sum(s_e) / (sum(s_e) + sum(s_x)), K_x = K - K_e.
    Clamp so each view gets at least 1 frame (if both have video).
    """
    se = max(sum(ego_scores), 1e-9)
    sx = max(sum(exo_scores), 1e-9)
    K_e = int(round(K * se / (se + sx)))
    K_e = max(1, min(K - 1, K_e))
    return K_e, K - K_e


# ── AKS selection ─────────────────────────────────────────────────────────────

def _meanstd_split(score_dicts, fns, n, t1, t2, all_depth):
    split_s, split_f       = [], []
    no_split_s, no_split_f = [], []
    for ds, fn in zip(score_dicts, fns):
        score = ds["score"]
        depth = ds["depth"]
        top_n     = heapq.nlargest(n, range(len(score)), score.__getitem__)
        mean_diff = np.mean([score[t] for t in top_n]) - np.mean(score)
        std       = np.std(score)
        if mean_diff > t1 and std > t2:
            no_split_s.append(ds); no_split_f.append(fn)
        elif depth < all_depth:
            mid = len(score) // 2
            split_s += [dict(score=score[:mid], depth=depth + 1),
                        dict(score=score[mid:], depth=depth + 1)]
            split_f += [fn[:mid], fn[mid:]]
        else:
            no_split_s.append(ds); no_split_f.append(fn)
    if split_s:
        rec_s, rec_f = _meanstd_split(split_s, split_f, n, t1, t2, all_depth)
    else:
        rec_s, rec_f = [], []
    return no_split_s + rec_s, no_split_f + rec_f


def aks_select(scores: list, k: int, t1: float, t2: float,
               all_depth: int) -> list:
    """Run AKS, return sorted list of selected 1FPS indices."""
    n = len(scores)
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    norm = np.array(scores, dtype=np.float32)
    lo, hi = norm.min(), norm.max()
    norm = (norm - lo) / (hi - lo) if hi > lo else np.zeros_like(norm)
    pos = list(range(n))
    segs, fns = _meanstd_split([dict(score=norm, depth=0)], [pos],
                                k, t1, t2, all_depth)
    out = []
    for seg, fn in zip(segs, fns):
        f_num = max(1, int(k / 2 ** seg["depth"]))
        topk  = heapq.nlargest(f_num, range(len(seg["score"])),
                                seg["score"].__getitem__)
        out.extend(fn[t] for t in topk)
    out = list(dict.fromkeys(out))  # deduplicate
    if len(out) > k:
        out.sort(key=lambda i: -norm[i])
        out = out[:k]
    out.sort()
    return out


# ── k-DPP selection ───────────────────────────────────────────────────────────

def kdpp_select(scores: list, vis_feats: np.ndarray, k: int) -> list:
    """
    Greedy MAP k-DPP on L_ij = s_i * (f_i · f_j) * s_j.
    vis_feats must be L2-normalised. Cholesky update, O(N*C*k).
    Returns sorted list of selected indices.
    """
    N = len(scores)
    if N == 0:
        return []
    if N <= k:
        return list(range(N))

    s = np.array(scores, dtype=np.float32)
    s = s - s.min()
    if s.max() > 0:
        s = s / s.max()

    # B = F * s[:, None],  L = B @ B.T
    B = vis_feats.astype(np.float32) * s[:, None]   # [N, C]
    d = np.einsum("ic,ic->i", B, B).copy()          # L_ii = ||B_i||^2

    selected = []
    V = np.zeros((N, k), dtype=np.float32)

    for t in range(k):
        if d.max() <= 1e-10:
            break
        j = int(np.argmax(d))
        selected.append(j)
        d[j] = -1.0

        if t < k - 1:
            L_col_j = B @ B[j]             # [N]  exact column of L
            if t > 0:
                L_col_j -= V[:, :t] @ V[j, :t]
            denom = np.sqrt(max(float(L_col_j[j]), 1e-10))
            V[:, t] = L_col_j / denom
            d -= V[:, t] ** 2
            np.maximum(d, 0.0, out=d)
            d[j] = -1.0

    return sorted(selected)


# ── cross-view k-DPP ──────────────────────────────────────────────────────────

def dpp_cross_select(ego_scores, ego_feats, exo_scores, exo_feats, k):
    """
    Joint greedy MAP k-DPP over ego ∪ exo pool.

    Joint kernel:
      L[i,j] = q_i * (f_i · f_j) * q_j   for all i,j in {ego ∪ exo}
    where cross-view blocks (ego-i, exo-j) use ego_feat_i · exo_feat_j,
    naturally penalising selecting similar frames across views.

    Returns (ego_sel, exo_sel) — sorted index lists into the original arrays.
    """
    N_e = len(ego_scores)
    N_x = len(exo_scores)
    N   = N_e + N_x
    if N == 0:
        return [], []
    if N <= k:
        return list(range(N_e)), list(range(N_x))

    # concatenate features and scores
    all_feats  = np.concatenate([ego_feats,  exo_feats],  axis=0).astype(np.float32)
    all_scores = np.array(list(ego_scores) + list(exo_scores), dtype=np.float32)

    # normalise quality scores to [0,1]
    all_scores = all_scores - all_scores.min()
    if all_scores.max() > 0:
        all_scores = all_scores / all_scores.max()

    # B_i = f_i * q_i  →  L_ij = B_i · B_j
    B = all_feats * all_scores[:, None]          # [N, C]
    d = np.einsum("ic,ic->i", B, B).copy()       # diagonal of L

    selected = []
    V = np.zeros((N, k), dtype=np.float32)

    for t in range(k):
        if d.max() <= 1e-10:
            break
        j = int(np.argmax(d))
        selected.append(j)
        d[j] = -1.0
        if t < k - 1:
            L_col_j = B @ B[j]
            if t > 0:
                L_col_j -= V[:, :t] @ V[j, :t]
            denom = np.sqrt(max(float(L_col_j[j]), 1e-10))
            V[:, t] = L_col_j / denom
            d -= V[:, t] ** 2
            np.maximum(d, 0.0, out=d)
            d[j] = -1.0

    ego_sel = sorted(i      for i in selected if i <  N_e)
    exo_sel = sorted(i - N_e for i in selected if i >= N_e)
    return ego_sel, exo_sel


# ── MDP3 (SigLIP + Seq-DPP + Multi-Gaussian kernel) ──────────────────────────

from torch import nn as _nn

class _MultiGaussianKernel(_nn.Module):
    def __init__(self, alphas=[2**k for k in range(-3, 2)]):
        super().__init__()
        self.alphas = alphas

    def forward(self, X, Y=None):
        Y = X.unsqueeze(0) if Y is None else Y.unsqueeze(0)
        X = X.unsqueeze(1)
        l2_sq = ((X - Y) ** 2).sum(2)
        return sum(torch.exp(-l2_sq / (2 * a)) for a in self.alphas)


_INF = 0x7fffffff

class _MDP3Selector:
    def __init__(self, device="cuda:0"):
        self.lamda          = 0.2
        self.segment_size   = 32
        self.condition_size = 1
        self.kernel         = _MultiGaussianKernel()
        self.device         = device
        self._model         = None
        self._proc          = None

    def load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, SiglipProcessor
        print(f"Loading SigLIP on {self.device} ...", flush=True)
        self._model = AutoModel.from_pretrained(
            SIGLIP_ID, torch_dtype=torch.float16, device_map=self.device).eval()
        self._proc  = SiglipProcessor.from_pretrained(SIGLIP_ID)
        self.kernel = self.kernel.to(self.device)
        print("SigLIP loaded.", flush=True)

    @torch.no_grad()
    def _embed(self, pils, prompt):
        proc, model = self._proc, self._model
        txt_inp  = proc(text=[prompt], truncation=True,
                        padding="max_length", return_tensors="pt").to(self.device)
        with torch.autocast(self.device.split(":")[0]):
            txt_feat = model.get_text_features(**txt_inp)
        txt_feat = (txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)).float()
        all_img  = []
        for i in range(0, len(pils), 64):
            vis_inp  = proc(images=pils[i:i+64], return_tensors="pt").to(self.device)
            with torch.autocast(self.device.split(":")[0]):
                img_feat = model.get_image_features(**vis_inp)
            img_feat = (img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)).float()
            all_img.append(img_feat)
            del vis_inp, img_feat; torch.cuda.empty_cache()
        return torch.cat(all_img, dim=0), txt_feat  # [N,C], [1,C]

    def _seqdpp(self, total_matrix, offset, to_select_num):
        if to_select_num == 0:
            return [0.0], [[]]
        cur_trace, ret_scores = [], [0.0]
        r, S = total_matrix[0:1, 1:], total_matrix[1:, 1:]
        cands = list(range(len(S) - offset))
        cond  = list(range(offset))
        L = None
        if cond:
            L = torch.linalg.cholesky(S[cond][:, cond])
        while len(cur_trace) < to_select_num:
            max_obj, cur_sel, better_L = -_INF, -1, None
            for i in cands:
                if i in cur_trace:
                    continue
                ci   = i + offset
                sidx = cond + [j + offset for j in cur_trace] + [ci]
                if L is None:
                    sv    = S[sidx][:, sidx]
                    cur_L = torch.sqrt(sv).reshape(1, 1)
                    logdet = sv.clone().log()
                else:
                    sv = S[ci:ci+1][:, sidx]
                    n  = L.shape[0]
                    sv = sv.view(-1, 1)
                    vp = torch.linalg.solve_triangular(L, sv[:n], upper=False)
                    nd = torch.sqrt(torch.abs(sv[-1] - vp.T @ vp))
                    cur_L = torch.zeros((n+1, n+1), dtype=L.dtype, device=L.device)
                    cur_L[:n, :n] = L
                    cur_L[n, :n]  = torch.cat([vp.flatten(), nd.view(1)])[:-1]
                    cur_L[n, n]   = nd
                    logdet = 2 * torch.log(torch.diag(cur_L)).sum()
                cur_obj = 1. / self.lamda * 2 * torch.log(r[:, sidx]).sum() + logdet
                if cur_obj > max_obj or cur_sel == -1:
                    max_obj, cur_sel, better_L = cur_obj, i, cur_L
            ret_scores.append(max_obj.clone() if isinstance(max_obj, torch.Tensor)
                               else torch.tensor(float(max_obj)))
            cur_trace.append(cur_sel)
            L = better_L
        traces = [sorted(cur_trace[:j]) for j in range(len(cur_trace) + 1)]
        return ret_scores, traces

    def select_indices(self, pils, prompt, k):
        N = len(pils)
        if N <= k:
            return list(range(N))
        self.segment_size = 32 if k <= 32 else 128
        image_embeds, text_embed = self._embed(pils, prompt)
        seg_size    = self.segment_size
        segment_num = (N + seg_size - 1) // seg_size
        dp    = [[0.] + [-_INF] * k for _ in range(segment_num + 1)]
        trace = [[[] for _ in range(k + 1)] for _ in range(segment_num + 1)]
        for seg_idx in range(1, segment_num + 1):
            cand_range = range((seg_idx-1)*seg_size, min(seg_idx*seg_size, N))
            cand_embs  = [image_embeds[i] for i in cand_range]
            sim_matrix = self.kernel(torch.stack(cand_embs))
            for start in range(0, min(k, (seg_idx-1)*seg_size) + 1):
                cond_idx = trace[seg_idx-1][start][
                    -min(self.condition_size, len(trace[seg_idx-1][start])):]
                offset   = len(cond_idx)
                add_embs = [text_embed[0].reshape(-1)] + [image_embeds[i] for i in cond_idx]
                additional = self.kernel(
                    torch.stack(add_embs),
                    torch.stack(add_embs + list(cand_embs)))
                total_matrix = torch.cat([
                    additional,
                    torch.cat([additional[:, -len(sim_matrix):].T, sim_matrix], dim=1)
                ], dim=0)
                max_sel = min(k - start, seg_size)
                cur_scores, cur_traces = self._seqdpp(total_matrix, offset, max_sel)
                for to_sel, (sc, tr) in enumerate(zip(cur_scores, cur_traces)):
                    tr_mapped = [i + int((seg_idx-1)*seg_size) for i in tr]
                    sc_total  = dp[seg_idx-1][start] + sc
                    tr_total  = trace[seg_idx-1][start] + tr_mapped
                    if sc_total > dp[seg_idx][start + to_sel]:
                        dp[seg_idx][start + to_sel]    = sc_total
                        trace[seg_idx][start + to_sel] = tr_total
        return sorted(trace[segment_num][k])


_mdp3_selector: _MDP3Selector = None


def load_mdp3(device):
    global _mdp3_selector
    _mdp3_selector = _MDP3Selector(device=device)
    _mdp3_selector.load()


def mdp3_sep_select(ego_pils, ego_scores, exo_pils, exo_scores, question, k):
    """Budget split by CLIP scores, then MDP3 Seq-DPP on each view, merge by timestamp."""
    if ego_pils and exo_pils:
        K_e, K_x = budget_split(ego_scores, exo_scores, k)
        ego_sel = _mdp3_selector.select_indices(ego_pils, question, K_e)
        exo_sel = _mdp3_selector.select_indices(exo_pils, question, K_x)
        return build_timeline(ego_pils, ego_sel, exo_pils, exo_sel)
    elif ego_pils:
        sel = _mdp3_selector.select_indices(ego_pils, question, k)
        return [(t, "ego", ego_pils[t]) for t in sel]
    else:
        sel = _mdp3_selector.select_indices(exo_pils, question, k)
        return [(t, "exo", exo_pils[t]) for t in sel]


def view_mdp3_select(ego_pils, ego_scores, exo_pils, exo_scores, question, k):
    """Per-timestamp pick ego or exo by CLIP score, then MDP3 Seq-DPP on the joint stream."""
    if ego_pils and exo_pils:
        aligned = view_select(ego_pils, ego_scores, exo_pils, exo_scores)
        if not aligned:
            return []
        pils = [pil for _, _, pil, _ in aligned]
        sel_idx = _mdp3_selector.select_indices(pils, question, k)
        return [(aligned[i][0], aligned[i][1], aligned[i][2]) for i in sel_idx]
    elif ego_pils:
        sel = _mdp3_selector.select_indices(ego_pils, question, k)
        return [(t, "ego", ego_pils[t]) for t in sel]
    else:
        sel = _mdp3_selector.select_indices(exo_pils, question, k)
        return [(t, "exo", exo_pils[t]) for t in sel]


# ── per-view selection ────────────────────────────────────────────────────────

def select_frames(scores, vis_feats, k, method, t1, t2, all_depth):
    """Dispatch to AKS or DPP."""
    if method == "aks":
        return aks_select(scores, k, t1, t2, all_depth)
    return kdpp_select(scores, vis_feats, k)


def build_timeline(ego_pils, ego_sel, exo_pils, exo_sel):
    """Return list of (timestamp, view, pil) sorted by timestamp."""
    frames = [(t, "ego", ego_pils[t]) for t in ego_sel]
    frames += [(t, "exo", exo_pils[t]) for t in exo_sel]
    frames.sort(key=lambda x: x[0])
    return frames


# ── BOLT view-select + inverse-transform sampling ─────────────────────────────

def bolt_sep_select(ego_pils, ego_scores, exo_pils, exo_scores, k):
    """
    bolt_sep: run ITS independently on ego and exo streams,
    budget split by mean CLIP score (same as AKS), then merge by timestamp.
    """
    if ego_pils and exo_pils:
        K_e, K_x = budget_split(ego_scores, exo_scores, k)
        ego_pool = [(t, "ego", ego_pils[t], float(ego_scores[t])) for t in range(len(ego_pils))]
        exo_pool = [(t, "exo", exo_pils[t], float(exo_scores[t])) for t in range(len(exo_pils))]
        ego_sel = its_select(ego_pool, K_e)
        exo_sel = its_select(exo_pool, K_x)
        merged = ego_sel + exo_sel
        merged.sort(key=lambda x: x[0])
        return merged
    elif ego_pils:
        pool = [(t, "ego", ego_pils[t], float(ego_scores[t])) for t in range(len(ego_pils))]
        return its_select(pool, k)
    else:
        pool = [(t, "exo", exo_pils[t], float(exo_scores[t])) for t in range(len(exo_pils))]
        return its_select(pool, k)


def view_select(ego_pils, ego_scores, exo_pils, exo_scores):
    """Per-timestamp: keep whichever view has higher CLIP score."""
    T = min(len(ego_pils), len(exo_pils))
    frames = []
    for t in range(T):
        if ego_scores[t] >= exo_scores[t]:
            frames.append((t, "ego", ego_pils[t], float(ego_scores[t])))
        else:
            frames.append((t, "exo", exo_pils[t], float(exo_scores[t])))
    return frames  # list of (ts, view, pil, score)


def view_dpp_select(ego_pils, ego_scores, ego_feats,
                    exo_pils, exo_scores, exo_feats, k):
    """
    Step 1: view_select — per-timestamp pick ego or exo by CLIP score.
    Step 2: k-DPP on the resulting single stream using the selected feats.
    Returns list of (ts, view, pil) sorted by timestamp.
    """
    aligned = view_select(ego_pils, ego_scores, exo_pils, exo_scores)
    N = len(aligned)
    if N == 0:
        return []

    # collect scores and feats for the selected stream
    scores = np.array([s for _, _, _, s in aligned], dtype=np.float32)
    feats  = np.stack([
        ego_feats[t] if v == "ego" else exo_feats[t]
        for t, v, _, _ in aligned
    ])  # [N, C]

    sel_idx = kdpp_select(scores.tolist(), feats, k)
    return [(aligned[i][0], aligned[i][1], aligned[i][2]) for i in sel_idx]


def its_select(frames_with_scores, k):
    """
    BOLT inverse-transform (systematic CDF) sampling.
    frames_with_scores: list of (ts, view, pil, score)
    Returns k frames sorted by timestamp.
    """
    N = len(frames_with_scores)
    if N <= k:
        return [(ts, v, p) for ts, v, p, _ in frames_with_scores]

    scores = np.array([s for _, _, _, s in frames_with_scores], dtype=np.float64)
    scores = scores - scores.min()
    if scores.sum() < 1e-12:
        scores = np.ones(N, dtype=np.float64)
    p   = scores / scores.sum()
    cdf = np.cumsum(p)

    quantiles = (np.arange(k) + 0.5) / k
    raw_idx   = np.searchsorted(cdf, quantiles)
    raw_idx   = np.clip(raw_idx, 0, N - 1)

    seen, selected = set(), []
    for i in raw_idx:
        if i not in seen:
            seen.add(i)
            selected.append(frames_with_scores[i])

    # fill gaps caused by duplicate CDF hits
    if len(selected) < k:
        for i, f in enumerate(frames_with_scores):
            if i not in seen:
                selected.append(f)
                seen.add(i)
                if len(selected) == k:
                    break

    selected.sort(key=lambda x: x[0])
    return [(ts, v, p) for ts, v, p, _ in selected]


# ── InternVL3.5 ───────────────────────────────────────────────────────────────

_internvl_model = None
_internvl_tok   = None
_iv_transform   = None


def _iv_build_transform(input_size=448):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _pil_to_iv(pil, input_size=448):
    global _iv_transform
    if _iv_transform is None:
        _iv_transform = _iv_build_transform(input_size)
    return _iv_transform(pil.convert("RGB")).unsqueeze(0)


def load_internvl(gpu_id):
    global _internvl_model, _internvl_tok
    if _internvl_model is not None:
        return
    from transformers import AutoModel, AutoTokenizer
    device = f"cuda:{gpu_id}"
    print(f"Loading InternVL3.5 on {device} ...", flush=True)
    try:
        _internvl_model = AutoModel.from_pretrained(
            LOCAL_INTERNVL_ID, torch_dtype=torch.bfloat16,
            use_flash_attn=True, trust_remote_code=True,
        ).eval().to(device)
    except Exception:
        _internvl_model = AutoModel.from_pretrained(
            LOCAL_INTERNVL_ID, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(device)
    _internvl_tok = AutoTokenizer.from_pretrained(
        LOCAL_INTERNVL_ID, trust_remote_code=True, use_fast=False)
    print("InternVL3.5 loaded.", flush=True)


def call_internvl(qa, frames):
    """frames: list of (timestamp, view, pil), pre-sorted by timestamp."""
    device = next(_internvl_model.parameters()).device
    pv_list, npatches, parts = [], [], []
    for idx, (ts, view, pil) in enumerate(frames, 1):
        pv = _pil_to_iv(pil).to(torch.bfloat16).to(device)
        pv_list.append(pv)
        npatches.append(pv.size(0))
        parts.append(f"[{view.capitalize()}, {ts}s] Frame{idx}: <image>")
    parts.append("\n" + build_text_prompt(qa))
    prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(parts)
    pixel_values = torch.cat(pv_list, dim=0)
    try:
        with torch.no_grad():
            raw = _internvl_model.chat(
                _internvl_tok, pixel_values, prompt,
                dict(max_new_tokens=16, do_sample=False),
                num_patches_list=npatches,
            )
        return parse_answer(raw), raw
    except Exception as e:
        return -1, str(e)
    finally:
        torch.cuda.empty_cache()


# ── Qwen2.5-VL ────────────────────────────────────────────────────────────────

_qwen_model = None
_qwen_proc  = None


def _resize_max(pil, max_side=480):
    w, h = pil.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), resample=3)
    return pil


def load_qwen(gpu_id):
    global _qwen_model, _qwen_proc
    if _qwen_model is not None:
        return
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    device = f"cuda:{gpu_id}"
    print(f"Loading Qwen2.5-VL on {device} ...", flush=True)
    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    _qwen_proc = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    print("Qwen2.5-VL loaded.", flush=True)


_llava_model = None
_llava_proc  = None


def load_llava(gpu_id):
    global _llava_model, _llava_proc
    if _llava_model is not None:
        return
    from transformers import AutoModelForCausalLM, AutoProcessor
    device = f"cuda:{gpu_id}"
    print(f"Loading LLaVA-OneVision on {device} ...", flush=True)
    _llava_model = AutoModelForCausalLM.from_pretrained(
        LLAVA_MODEL_ID, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    ).eval()
    _llava_proc = AutoProcessor.from_pretrained(LLAVA_MODEL_ID, trust_remote_code=True)
    print("LLaVA-OneVision loaded.", flush=True)


def call_llava(qa, frames):
    """frames: list of (timestamp, view, pil), pre-sorted by timestamp."""
    from qwen_vl_utils import process_vision_info
    device = next(_llava_model.parameters()).device
    content = []
    for ts, view, pil in frames:
        content.append({"type": "text", "text": f"[{view.capitalize()}, {ts}s]"})
        content.append({"type": "image", "image": _resize_max(pil, 480)})
    content.append({"type": "text", "text": build_text_prompt(qa)})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    try:
        text = _llava_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = _llava_proc(
            text=[text], images=image_inputs, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = _llava_model.generate(**inputs, max_new_tokens=16, do_sample=False)
        out = out[:, inputs["input_ids"].shape[-1]:]
        raw = _llava_proc.batch_decode(out, skip_special_tokens=True)[0]
        return parse_answer(raw), raw
    except Exception as e:
        return -1, str(e)
    finally:
        torch.cuda.empty_cache()


# ── Qwen3-VL ─────────────────────────────────────────────────────────────────

_qwen3vl_model = None
_qwen3vl_proc  = None


def load_qwen3vl(gpu_id):
    global _qwen3vl_model, _qwen3vl_proc
    if _qwen3vl_model is not None:
        return
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    device = f"cuda:{gpu_id}"
    print(f"Loading Qwen3-VL on {device} ...", flush=True)
    _qwen3vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN3VL_MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    _qwen3vl_proc = AutoProcessor.from_pretrained(QWEN3VL_MODEL_ID)
    print("Qwen3-VL loaded.", flush=True)


def call_qwen3vl(qa, frames):
    """frames: list of (timestamp, view, pil), pre-sorted by timestamp."""
    from qwen_vl_utils import process_vision_info
    device = next(_qwen3vl_model.parameters()).device
    content = []
    for ts, view, pil in frames:
        content.append({"type": "text", "text": f"[{view.capitalize()}, {ts}s]"})
        content.append({"type": "image", "image": _resize_max(pil, 480)})
    content.append({"type": "text", "text": build_text_prompt(qa)})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    try:
        text = _qwen3vl_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        image_inputs, _ = process_vision_info(messages)
        inputs = _qwen3vl_proc(
            text=[text], images=image_inputs, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = _qwen3vl_model.generate(**inputs, max_new_tokens=16, do_sample=False)
        out = out[:, inputs["input_ids"].shape[-1]:]
        raw = _qwen3vl_proc.batch_decode(out, skip_special_tokens=True)[0]
        return parse_answer(raw), raw
    except Exception as e:
        return -1, str(e)
    finally:
        torch.cuda.empty_cache()


def call_qwen(qa, frames):
    """frames: list of (timestamp, view, pil), pre-sorted by timestamp."""
    from qwen_vl_utils import process_vision_info
    device = next(_qwen_model.parameters()).device
    content = []
    for ts, view, pil in frames:
        content.append({"type": "text", "text": f"[{view.capitalize()}, {ts}s]"})
        content.append({"type": "image", "image": _resize_max(pil, 480)})
    content.append({"type": "text", "text": build_text_prompt(qa)})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]
    try:
        text = _qwen_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = _qwen_proc(
            text=[text], images=image_inputs, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = _qwen_model.generate(**inputs, max_new_tokens=16, do_sample=False)
        out = out[:, inputs["input_ids"].shape[-1]:]
        raw = _qwen_proc.batch_decode(out, skip_special_tokens=True)[0]
        return parse_answer(raw), raw
    except Exception as e:
        return -1, str(e)
    finally:
        torch.cuda.empty_cache()


# ── Gemini 2.5-flash ─────────────────────────────────────────────────────────

_gemini_client = None


def load_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return
    from google import genai
    _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print("Gemini client loaded.", flush=True)


def call_gemini(qa, frames):
    """frames: list of (timestamp, view, pil), pre-sorted by timestamp."""
    import io
    from google.genai import types
    parts = [types.Part.from_text(text=SYSTEM_PROMPT)]
    for ts, view, pil in frames:
        parts.append(types.Part.from_text(text=f"[{view.capitalize()}, {ts}s]"))
        buf = io.BytesIO()
        _resize_max(pil, 480).save(buf, format="JPEG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=build_text_prompt(qa)))
    try:
        resp = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=types.Content(role="user", parts=parts),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = resp.text.strip()
        return parse_answer(raw), raw
    except Exception as e:
        print(f"    Gemini error: {e}", flush=True)
        return -1, str(e)


# ── shared QA helpers ─────────────────────────────────────────────────────────

def build_text_prompt(qa):
    letters = "ABCD"
    lines = [f"Question: {qa['question']}", "Options:"]
    for i, o in enumerate(qa["options"]):
        lines.append(f"  {letters[i]}. {o}")
    lines.append("\nAnswer with the option letter only (A, B, C, or D).")
    return "\n".join(lines)


def parse_answer(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\b([A-D])\b", raw, re.IGNORECASE)
    return ord(m.group(1).upper()) - ord("A") if m else -1


# ── dataset loaders ───────────────────────────────────────────────────────────

def shuffle_options(qa):
    rng = random.Random(qa["qa_id"])
    options = list(qa["options"])
    correct = options[qa["correct_answer"]]
    rng.shuffle(options)
    return {**qa, "options": options, "correct_answer": options.index(correct)}


def load_egoexo_multi():
    with open(EGOEXO_QA) as f:
        raw = json.load(f)
    return [shuffle_options({
        "qa_id": q["qa_id"], "qa_type": q["qa_type"],
        "question": q["question"], "options": q["options"],
        "correct_answer": q["correct_answer"],
        "ego_path": get_video_path("egoexo", q["take_name"], "ego"),
        "exo_path": get_video_path("egoexo", q["take_name"], "exo"),
        "take_name": q["take_name"], "dataset": "egoexo",
    }) for q in raw]


def load_lemma_multi():
    with open(LEMMA_QA) as f:
        raw = json.load(f)
    def get_type(qa_id):
        for p in qa_id.split("_"):
            if p.startswith("Q") and p[1:].isdigit():
                return p
        return "?"
    out = []
    for qa_id, q in raw.items():
        if q.get("status") != "ok":
            continue
        take = lemma_take_name(qa_id)
        out.append(shuffle_options({
            "qa_id": qa_id, "qa_type": get_type(qa_id),
            "question": q["question"], "options": q["options"],
            "correct_answer": q["correct_answer"],
            "ego_path": get_video_path("lemma", take, "ego"),
            "exo_path": get_video_path("lemma", take, "exo"),
            "take_name": take, "dataset": "lemma",
        }))
    return out


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(qa_list, cache, cache_path, device, method, model,
             K=TOTAL_FRAMES, t1=T1, t2=T2, all_depth=ALL_DEPTH,
             save_every=10, views=("ego", "exo")):
    if model == "qwen":
        call_vlm = call_qwen
    elif model == "llava":
        call_vlm = call_llava
    elif model == "gemini":
        call_vlm = call_gemini
    elif model == "qwen3vl":
        call_vlm = call_qwen3vl
    else:
        call_vlm = call_internvl

    correct_by_type = defaultdict(int)
    total_by_type   = defaultdict(int)
    no_data    = 0
    since_save = 0

    for i, qa in enumerate(qa_list):
        cache_key = f"egoexomem_{method}_{model}_{qa['qa_id']}"
        was_cached = cache_key in cache

        if cache_key not in cache:
            ego_vid = qa.get("ego_path") if "ego" in views else None
            exo_vid = qa.get("exo_path") if "exo" in views else None

            ego_pils = (extract_frames_1fps(Path(ego_vid))[1]
                        if ego_vid and Path(str(ego_vid)).exists() else [])
            exo_pils = (extract_frames_1fps(Path(exo_vid))[1]
                        if exo_vid and Path(str(exo_vid)).exists() else [])

            if not ego_pils and not exo_pils:
                cache[cache_key] = {"predicted": -1, "no_video": True}
                no_data += 1
                continue

            question = qa["question"]

            ego_scores, ego_feats = (
                clip_score_and_features(ego_pils, question, device)
                if ego_pils else ([], np.zeros((0, 512), dtype=np.float32)))
            exo_scores, exo_feats = (
                clip_score_and_features(exo_pils, question, device)
                if exo_pils else ([], np.zeros((0, 512), dtype=np.float32)))

            # budget + selection
            if method == "bolt":
                if ego_pils and exo_pils:
                    aligned = view_select(ego_pils, ego_scores, exo_pils, exo_scores)
                    frames  = its_select(aligned, K)
                elif ego_pils:
                    pool   = [(t, "ego", ego_pils[t], float(ego_scores[t]))
                              for t in range(len(ego_pils))]
                    frames = its_select(pool, K)
                else:
                    pool   = [(t, "exo", exo_pils[t], float(exo_scores[t]))
                              for t in range(len(exo_pils))]
                    frames = its_select(pool, K)
            elif method == "bolt_sep":
                frames = bolt_sep_select(ego_pils, ego_scores, exo_pils, exo_scores, K)
            elif method == "dpp_cross" and ego_pils and exo_pils:
                ego_sel, exo_sel = dpp_cross_select(
                    ego_scores, ego_feats, exo_scores, exo_feats, K)
                frames = build_timeline(ego_pils, ego_sel, exo_pils, exo_sel)
            elif method == "view_dpp" and ego_pils and exo_pils:
                frames = view_dpp_select(
                    ego_pils, ego_scores, ego_feats,
                    exo_pils, exo_scores, exo_feats, K)
            elif method == "mdp3_sep":
                frames = mdp3_sep_select(
                    ego_pils, ego_scores, exo_pils, exo_scores, question, K)
            elif method == "view_mdp3":
                frames = view_mdp3_select(
                    ego_pils, ego_scores, exo_pils, exo_scores, question, K)
            elif ego_pils and exo_pils:
                K_e, K_x = budget_split(ego_scores, exo_scores, K)
                ego_sel = select_frames(ego_scores, ego_feats, K_e, method, t1, t2, all_depth)
                exo_sel = select_frames(exo_scores, exo_feats, K_x, method, t1, t2, all_depth)
                frames = build_timeline(ego_pils, ego_sel, exo_pils, exo_sel)
            elif ego_pils:
                ego_sel = select_frames(ego_scores, ego_feats, K, method, t1, t2, all_depth)
                frames = [(t, "ego", ego_pils[t]) for t in ego_sel]
            else:
                exo_sel = select_frames(exo_scores, exo_feats, K, method, t1, t2, all_depth)
                frames = [(t, "exo", exo_pils[t]) for t in exo_sel]

            pred, raw = call_vlm(qa, frames)

            n_ego = sum(1 for _, v, _ in frames if v == "ego")
            n_exo = sum(1 for _, v, _ in frames if v == "exo")
            frames_str = f"ego={n_ego},exo={n_exo}"

            is_correct = pred == qa["correct_answer"]
            cache[cache_key] = {
                "qa_id":        qa["qa_id"],
                "qa_type":      qa["qa_type"],
                "dataset":      qa["dataset"],
                "view":         "multi",
                "take_name":    qa["take_name"],
                "question":     qa["question"],
                "options":      qa["options"],
                "n_frames":     len(frames),
                "predicted":    pred,
                "pred_text":    qa["options"][pred] if 0 <= pred < len(qa["options"]) else None,
                "correct":      qa["correct_answer"],
                "correct_text": qa["options"][qa["correct_answer"]],
                "is_correct":   is_correct,
                "response":     raw,
            }
            mark = "✓" if is_correct else "✗"
            print(f"  [{i+1}/{len(qa_list)}] {mark} {qa['qa_type']} "
                  f"pred={pred} gt={qa['correct_answer']} | {qa['qa_id']} [{frames_str}]",
                  flush=True)
            print(f"    response: {raw}", flush=True)
            time.sleep(0.05)

            since_save += 1
            if since_save >= save_every:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                since_save = 0

        entry = cache[cache_key]
        if entry.get("no_video"):
            print(f"  [{i+1}/{len(qa_list)}] NO_VIDEO {qa['qa_type']} | {qa['qa_id']}",
                  flush=True)
            continue
        if was_cached and "is_correct" in entry:
            mark = "✓" if entry["is_correct"] else "✗"
            print(f"  [{i+1}/{len(qa_list)}] {mark} {entry['qa_type']} "
                  f"pred={entry['predicted']} gt={entry['correct']} | "
                  f"{entry['qa_id']} [cached]", flush=True)

        qt = qa["qa_type"]
        total_by_type[qt]   += 1
        if entry.get("is_correct"):
            correct_by_type[qt] += 1

    if no_data:
        print(f"  Skipped {no_data} (no video)", flush=True)

    total   = sum(total_by_type.values())
    correct = sum(correct_by_type.values())
    result  = {"overall": {"correct": correct, "total": total,
                            "acc": correct / total if total else 0.0}}
    for qt in sorted(total_by_type):
        n, c = total_by_type[qt], correct_by_type[qt]
        result[qt] = {"correct": c, "total": n, "acc": c / n if n else 0.0}
    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",   default="aks", choices=["aks", "dpp", "dpp_cross", "view_dpp", "bolt", "bolt_sep", "mdp3_sep", "view_mdp3"])
    parser.add_argument("--model",    default="internvl", choices=["internvl", "qwen", "llava", "gemini", "qwen3vl"])
    parser.add_argument("--datasets", nargs="+", default=["egoexo", "lemma"],
                        choices=["egoexo", "lemma"])
    parser.add_argument("--views",    nargs="+", default=["ego", "exo"],
                        choices=["ego", "exo"])
    parser.add_argument("--out",      default=None)
    parser.add_argument("--gpu-id",   type=int, default=0)
    parser.add_argument("--K",        type=int, default=TOTAL_FRAMES,
                        help="Total frame budget (default: 32)")
    parser.add_argument("--t1",       type=float, default=T1)
    parser.add_argument("--t2",       type=float, default=T2)
    parser.add_argument("--all-depth",type=int,   default=ALL_DEPTH)
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--shard",       type=int, default=0)
    parser.add_argument("--total-shards",type=int, default=1)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    load_clip(device)
    if args.method in ("mdp3_sep", "view_mdp3"):
        load_mdp3(device)
    if args.model == "qwen":
        load_qwen(args.gpu_id)
    elif args.model == "llava":
        load_llava(args.gpu_id)
    elif args.model == "gemini":
        load_gemini_client()
    elif args.model == "qwen3vl":
        load_qwen3vl(args.gpu_id)
    else:
        load_internvl(args.gpu_id)

    qa_all = []
    for ds in args.datasets:
        qa_all += load_egoexo_multi() if ds == "egoexo" else load_lemma_multi()
    if args.limit:
        qa_all = qa_all[:args.limit]
    if args.total_shards > 1:
        qa_all = [q for i, q in enumerate(qa_all) if i % args.total_shards == args.shard]

    out_dir = Path(args.out) if args.out else \
              Path(f"results_rag/egoexo_{args.method}_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "eval_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    already = sum(1 for v in cache.values() if "is_correct" in v)

    print(f"\n=== EgoExo-{args.method.upper()} | {args.model} | "
          f"K={args.K} | questions={len(qa_all)} | cached={already} ===", flush=True)

    result = evaluate(qa_all, cache, cache_path, device,
                      method=args.method, model=args.model,
                      K=args.K, t1=args.t1, t2=args.t2, all_depth=args.all_depth,
                      views=args.views)

    with open(cache_path, "w") as f:
        json.dump(cache, f)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    total, correct = result["overall"]["total"], result["overall"]["correct"]
    print(f"\nOverall: {correct}/{total} = {correct/total*100:.2f}%", flush=True)
    qs = sorted(k for k in result if k != "overall")
    for qt in qs:
        r = result[qt]
        print(f"  {qt}: {r['correct']}/{r['total']} = {r['acc']*100:.2f}%", flush=True)
    qavg = sum(result[q]["acc"] for q in qs) / len(qs)
    print(f"Qavg: {qavg*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()

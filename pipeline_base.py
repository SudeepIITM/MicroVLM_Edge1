import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

FRAMES = int(os.getenv("FRAMES", "8"))

def sample_frames(path, k=FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in range(k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / k))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    cap.release()
    if not frames:
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(k)]
    elif len(frames) < k:
        last_frame = frames[-1]
        frames.extend([last_frame] * (k - len(frames)))
    elif len(frames) > k:
        frames = frames[:k]
    return frames

class Chomp1d(nn.Module):
    """Removes padding to ensure causal convolution."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """Temporal convolution block with residual connection."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        return out.transpose(1, 2)

class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, adapter_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, adapter_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(adapter_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x

class EnhancedTemporalAdapter(nn.Module):
    def __init__(self, in_dim, proj_dim, nhead, adapter_dim, tcn_layers, attn_layers, dropout, max_len=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        pe = torch.zeros(max_len, proj_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, proj_dim, 2).float() * (-math.log(10000.0) / proj_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.tcn = TemporalConvNet(
            num_inputs=proj_dim,
            num_channels=[proj_dim] * tcn_layers,
            kernel_size=3,
            dropout=dropout
        )
        self.attn_blocks = nn.ModuleList([
            TemporalAttentionBlock(proj_dim, nhead, adapter_dim, dropout)
            for _ in range(attn_layers)
        ])
        self.norm = nn.LayerNorm(proj_dim)
        self.scale_fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x[:, 1:, :] = x[:, 1:, :] + self.pe[:, :T, :]
        tcn_out = self.tcn(x)
        attn_out = x
        for blk in self.attn_blocks:
            attn_out = blk(attn_out)
        combined = torch.cat([tcn_out[:, 0, :], attn_out[:, 0, :]], dim=1)
        return self.scale_fusion(combined)

class CrossAttentionFusion(nn.Module):
    def __init__(self, proj_dim, nhead, dropout):
        super().__init__()
        self.norm_v = nn.LayerNorm(proj_dim)
        self.norm_t = nn.LayerNorm(proj_dim)
        self.cross_attn = nn.MultiheadAttention(proj_dim, nhead, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(proj_dim)
        self.ff = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim * 4, proj_dim), nn.Dropout(dropout),
        )

    def forward(self, visual, text):
        v_seq = visual.unsqueeze(1)
        t_seq = text.unsqueeze(1)
        v_out, _ = self.cross_attn(self.norm_v(v_seq), self.norm_t(t_seq), self.norm_t(t_seq))
        v_out = v_out.squeeze(1)
        t_out, _ = self.cross_attn(self.norm_t(t_seq), self.norm_v(v_seq), self.norm_v(v_seq))
        t_out = t_out.squeeze(1)
        fused = self.norm_out(v_out + visual + t_out)
        fused = fused + self.ff(fused)
        return fused

class GatedFusion(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid()
        )

    def forward(self, visual, text):
        concat = torch.cat([visual, text], dim=1)
        gate = self.gate(concat)
        return gate * visual + (1 - gate) * text

class TaskSpecificQueryAdapter(nn.Module):
    def __init__(self, proj_dim, n_queries_per_task, nhead, dropout):
        super().__init__()
        self.n_queries = n_queries_per_task
        tasks = ["action_ml", "action_fine", "action_sup", "weapon", "location", "people"]
        self.task_queries = nn.ParameterDict({
            task: nn.Parameter(torch.randn(1, n_queries_per_task, proj_dim))
            for task in tasks
        })
        self.cross_attn = nn.ModuleDict({
            task: nn.MultiheadAttention(proj_dim, nhead, dropout=dropout, batch_first=True)
            for task in tasks
        })
        self.norm = nn.ModuleDict({
            task: nn.LayerNorm(proj_dim) for task in tasks
        })
        for task in tasks:
            nn.init.trunc_normal_(self.task_queries[task], std=0.02)

    def forward(self, features, task_name):
        B = features.shape[0]
        q = self.task_queries[task_name].expand(B, -1, -1)
        f_seq = features.unsqueeze(1)
        attn_out, _ = self.cross_attn[task_name](q, f_seq, f_seq)
        attn_out = self.norm[task_name](attn_out + q)
        return attn_out.mean(dim=1)

class EnhancedTemporalAdapterModel(nn.Module):
    def __init__(self, in_dim, proj_dim, nhead, adapter_dim, tcn_layers, attn_layers, dropout,
                 n_ml, n_fine, n_sup, n_wpn, n_loc, n_ppl, n_queries=4, max_len=64):
        super().__init__()
        self.adapter = EnhancedTemporalAdapter(in_dim, proj_dim, nhead, adapter_dim,
                                                tcn_layers, attn_layers, dropout, max_len=max_len)
        self.text_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attn_fusion = CrossAttentionFusion(proj_dim, nhead, dropout)
        self.gated_fusion = GatedFusion(proj_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.task_query_adapter = TaskSpecificQueryAdapter(proj_dim, n_queries, nhead, dropout)
        self.heads = nn.ModuleDict({
            "action_ml": nn.Linear(proj_dim, n_ml),
            "action_fine": nn.Linear(proj_dim, n_fine),
            "action_sup": nn.Linear(proj_dim, n_sup),
            "weapon": nn.Linear(proj_dim, n_wpn),
            "location": nn.Linear(proj_dim, n_loc),
            "people": nn.Linear(proj_dim, n_ppl),
        })

    def forward(self, frames, text_emb):
        visual_cls = self.adapter(frames)
        text_proj = self.text_proj(text_emb)
        cross_fused = self.cross_attn_fusion(visual_cls, text_proj)
        gated_fused = self.gated_fusion(visual_cls, text_proj)
        combined = torch.cat([cross_fused, gated_fused], dim=1)
        fused = self.fusion_proj(combined)
        return {
            "action_ml": self.heads["action_ml"](self.task_query_adapter(fused, "action_ml")),
            "action_fine": self.heads["action_fine"](self.task_query_adapter(fused, "action_fine")),
            "action_sup": self.heads["action_sup"](self.task_query_adapter(fused, "action_sup")),
            "weapon": self.heads["weapon"](self.task_query_adapter(fused, "weapon")),
            "location": self.heads["location"](self.task_query_adapter(fused, "location")),
            "people": self.heads["people"](self.task_query_adapter(fused, "people")),
        }

def embed_frames(frames):
    embs = []
    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        txt = qwen_processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Represent this surveillance video frame for crime activity retrieval."},
            ]}], tokenize=False, add_generation_prompt=False,
        )
        inp = qwen_processor(text=[txt], images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = qwen_model(**inp, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].squeeze(0)
            h = F.normalize(h.float(), dim=-1)
        embs.append(h.cpu())
    return torch.stack(embs, dim=0)

def embed_text(text):
    txt = qwen_processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "text", "text": f"Represent this video summary for crime activity retrieval.\n\n{text}"},
        ]}], tokenize=False, add_generation_prompt=False,
    )
    inp = qwen_processor(text=[txt], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = qwen_model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :].squeeze(0)
        h = F.normalize(h.float(), dim=-1)
    return h.cpu()

def generate_summary(frames, max_new_tokens=60):
    """Generate a real summary from the sampled frames using the summarization model."""
    images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what is happening in this surveillance video in 2-3 sentences."},
            *[{"type": "image"} for _ in images],
        ],
    }]
    prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(images=images, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = summary_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = ids[:, inputs["input_ids"].shape[1]:]
    text = qwen_processor.decode(new_ids[0], skip_special_tokens=True)
    return text, new_ids.shape[1]

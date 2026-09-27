import torch
import torch.nn.functional as F
import dgl.nn.pytorch as graph
import numpy as np
import torch.nn as nn
from torch.nn import Parameter, Linear, ModuleList


# ==================== 以下辅助类保持不变 ====================

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=F.elu):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else lambda x: x

    def forward(self, embeds, nty):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = sum(embeds[i] * beta[i] for i in range(len(embeds)))
        return z_mc, beta.data.cpu().numpy()


class HeteroGraphConv(nn.Module):
    def __init__(self, mods, aggregate='att', hidden_dim=64, attn_drop=0.0):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(fn):
                fn(True)
        if aggregate == 'att':
            self.agg_fn = inter_att(hidden_dim, attn_drop)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}

        if isinstance(inputs, tuple) or g.is_block:
            src_inputs, dst_inputs = inputs if isinstance(inputs, tuple) else (
                inputs,
                {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
            )
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph, (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()), **mod_kwargs.get(etype, {})
                )
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph, (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()), **mod_kwargs.get(etype, {})
                )
                outputs[dtype].append(dstdata)

        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty], _ = self.agg_fn(alist, nty)
        return rsts


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class AdaptiveLowRankProjection(nn.Module):
    """
    统一版低秩投影（bottleneck + 加法残差），适用于所有数据集。
    返回 (out, gate_w)，第二项与旧版 scalar 的 eta_k 形状兼容，无需修改调用代码。
    """
    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # 瓶颈投影（低秩重建）
        self.proj_down = nn.Linear(hidden_dim, rank, bias=False)
        self.proj_up   = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.orthogonal_(self.proj_up.weight)
        # 门控：控制低秩修正注入强度
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[-2].bias, -2.0)
        # 残差变换（主干）
        self.residual_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.last_gate_mean = 0.0

    def forward(self, x, gamma_k):
        # 主干（永远保留）
        x_res = self.residual_transform(x)

        # 输出门控（节点自适应注入强度）
        gate_w = self.gate(x)                     # (N, 1)
        self.last_gate_mean = gate_w.detach().mean().item()

        # 瓶颈通路：降维 → 通道加权（gamma_k）→ 升维 → 非线性
        h_k  = torch.tanh(self.proj_down(x))      # (N, rank)
        h_k  = h_k * gamma_k.t()                  # (N, rank)
        #x_lr = self.proj_up(h_k)
        x_lr = torch.tanh(self.proj_up(h_k))      # (N, hidden_dim)

        # 加法残差融合（低秩修正叠加到主干）
        out = x_res + gate_w * x_lr

        # 返回两个值，第二项为 (N,1) 张量，与原 eta_k 形状一致，解包不会报错
        return out, gate_w

# ==================== 修改后的主模型 ====================

class HeteroGraphSAGE(nn.Module):
    def __init__(self, g, in_feats, hidden_feats, out_feats, num_layers,
                 aggregator_type_dict, featsdim, topo_dim,
                 dataset, alpha, Init, rank, K, attn_drop=0.0,beta=0.3,
                 zpre_dim=None):
        super(HeteroGraphSAGE, self).__init__()

        self.rank = rank
        self.alpha = alpha
        self.K = K
        self.hidden_feats = hidden_feats
        self.beta = beta            # 初始残差强度

        # ========== 低秩参数初始化 ==========
        if Init == 'PPR':
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
            TEMP = torch.tensor(np.array([TEMP for _ in range(rank)]))
        elif Init == 'Random':
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
            TEMP = torch.tensor(np.array([TEMP for _ in range(rank)]))
        elif Init == 'Fix':
            TEMP = np.ones(K + 1)
            TEMP = torch.tensor(np.array([TEMP for _ in range(rank)]))
        elif Init == 'Mine':
            para = torch.ones([rank, K + 1])
            TEMP = torch.nn.init.xavier_normal_(para)
        elif Init == 'Mine_PPR':
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
            TEMP = torch.tensor(np.array([TEMP] * rank))
        else:
            raise ValueError(f"Unknown Init type: {Init}")

        self.gamma = Parameter(TEMP.float())

        # ========== 独立低秩增强模块（每阶一个） ==========
        self.adaptive_proj_list = ModuleList([
            AdaptiveLowRankProjection(hidden_feats, rank) for _ in range(K + 1)
        ])

        self.proj_h = MLPProjector(hidden_feats, hidden_feats)
        self.proj_z = MLPProjector(hidden_feats, hidden_feats)

        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=0.5)
        self.dataset = dataset
        self.decoder = InnerProductDecoder(0.5, act=lambda x: x)

        self.layers = nn.ModuleDict({
            ntype: nn.Linear(in_feats[ntype], hidden_feats)
            for ntype in aggregator_type_dict
        })

        # 预测层：动态适配预训练嵌入维度
        if zpre_dim is None:
            zpre_dim = hidden_feats * 2
        self.predict3 = nn.Linear(zpre_dim, hidden_feats, bias=True)

        # ========== 只需要一个异构图卷积算子（幂迭代） ==========
        conv_mods = {rel: graph.GraphConv(hidden_feats, hidden_feats) for rel in g.etypes}
        self.conv = HeteroGraphConv(conv_mods, aggregate='att',
                                    hidden_dim=hidden_feats, attn_drop=attn_drop)

        self.predict1 = nn.Sequential(
            nn.Linear(hidden_feats, 128), nn.ReLU(),
            nn.Linear(128, featsdim), nn.Tanh()
        )
        self.predict4 = nn.Sequential(
            nn.Linear(hidden_feats, 128), nn.ReLU(),
            nn.Linear(128, featsdim), nn.Tanh()
        )
        self.out_layer = nn.Linear(hidden_feats, out_feats)
        self.mask_token = nn.Parameter(torch.empty(1, featsdim))
        nn.init.xavier_normal_(self.mask_token)
        self.predictor_z = nn.Linear(hidden_feats, hidden_feats)

    def forward(self, g, ADJ_TOPO, z_pre, features_v1, target, tau, w_dis, kl2_weight=1.0):
        # 输入处理
        g.ndata['feature'][target] = features_v1
        h_dict = {ntp: self.dropout(self.layers[ntp](g.ndata['feature'][ntp]))
                  for ntp in g.ntypes}

        # ── 幂迭代 + 初始残差：生成各阶 target 节点表示 ──
        prop_results = [h_dict[target]]          # 第0阶

        # 第1阶：从初始特征出发做一次异构卷积
        cur_dict = self.conv(g, h_dict)
        prop_results.append(cur_dict[target])

        # 后续高阶：反复施加同一算子，并混合初始特征（初始残差）
        for k in range(1, self.K):
            next_dict = self.conv(g, cur_dict)
            # 初始残差（GCNII / APPNP 风格）
            for ntype in next_dict:
                if ntype in h_dict:
                    next_dict[ntype] = (1-self.beta) * next_dict[ntype] + self.beta * h_dict[ntype]
            prop_results.append(next_dict[target])
            cur_dict = next_dict

        # ── 独立低秩增强 + 加权求和 ──
        hidden = 0
        for k, h_k in enumerate(prop_results):
            gamma_k = self.gamma[:, k].unsqueeze(-1)           # (rank, 1)
            h_k_enhanced, _ = self.adaptive_proj_list[k](h_k, gamma_k)
            hidden = hidden + h_k_enhanced
        #hidden=sum(prop_results)
        h = torch.tanh(hidden)

        # ========== 以下蒸馏、预测等完全不变 ==========
        z_fused = self.predict3(z_pre)
        ADJ_P = self.decoder(h)
        x_pre = self.predict4(h)
        topo_pre = self.predict1(h)
        z_pre1 = self.predict1(z_fused)

        N = h.shape[0]
        h_p = F.normalize(self.proj_h(h), dim=-1)
        z_p = F.normalize(self.predictor_z(self.proj_z(z_fused)), dim=-1)

        if not ADJ_TOPO.is_sparse:
            ADJ_TOPO = ADJ_TOPO.to_sparse_coo()
        row, col = ADJ_TOPO.indices()
        mask = (row < N) & (col < N)
        row, col = row[mask], col[mask]
        sl = torch.arange(N, device=h.device)
        row = torch.cat([row, sl])
        col = torch.cat([col, sl])

        sim_h = (h_p[row] * h_p[col]).sum(-1).clamp(-50, 50) / tau
        sim_z = (z_p[row] * z_p[col]).sum(-1).clamp(-50, 50) / tau

        def sparse_softmax(sim, row, N):
            row = row.clamp(0, N - 1)
            mx = torch.full((N,), float('-inf'), device=sim.device)
            mx = torch.nan_to_num(mx.scatter_reduce(0, row, sim, reduce='amax'))
            exp = torch.exp(sim - mx[row])
            s = torch.zeros(N, device=sim.device).scatter_add(0, row, exp)
            p = exp / (s[row] + 1e-12)
            return p, torch.log(p + 1e-12)

        p_h, log_p_h = sparse_softmax(sim_h, row, N)
        p_z, log_p_z = sparse_softmax(sim_z, row, N)
        loss_dis = w_dis * (-(p_z.detach() * log_p_h).mean()
                            + kl2_weight * -(p_h.detach() * log_p_z).mean())
        #loss_dis = torch.tensor(0.0, device=h.device)

        return self.out_layer(self.dropout(h)), h, x_pre, z_pre1, ADJ_P, loss_dis, topo_pre

    def reset_parameters(self):
        torch.nn.init.zeros_(self.gamma)
        for k in range(self.K + 1):
            self.gamma.data[k] = self.alpha * (1 - self.alpha) ** k
        self.gamma.data[-1] = (1 - self.alpha) ** self.K

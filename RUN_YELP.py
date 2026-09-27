import argparse
import os
import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp

from load_data import load_data_YELP  # Yelp 数据加载
from model import HeteroGraphSAGE  # ALRE 增强模型
from tools import evaluate_results_nc, EarlyStopping


# ==================== 工具函数 ====================

def compute_topo_target(g, target, x_true, device):
    """计算一阶邻居聚合特征作为拓扑目标"""
    N = g.num_nodes(target)
    feat_dim = x_true.shape[1]
    agg = torch.zeros(N, feat_dim, device=device)
    cnt = torch.zeros(N, 1, device=device)

    for stype, etype, dtype in g.canonical_etypes:
        if dtype != target:
            continue
        src, dst = g[stype, etype, dtype].edges()
        src = src.to(device).long()
        dst = dst.to(device).long()

        if stype == target:
            src_feat = x_true[src]
        else:
            raw = g.ndata['feature'][stype].to(device)
            if raw.shape[1] >= feat_dim:
                src_feat = raw[src, :feat_dim]
            else:
                pad = torch.zeros(src.shape[0], feat_dim - raw.shape[1], device=device)
                src_feat = torch.cat([raw[src], pad], dim=1)

        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, feat_dim), src_feat)
        cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0], 1, device=device))

    topo_true = agg / cnt.clamp(min=1.0)
    return topo_true


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_activation_weights(features):
    """计算特征的激活频率权重（用于自适应掩码）"""
    activation_freq = (features != 0).float().mean(dim=0)
    mean_freq = activation_freq.mean().clamp(min=1e-8)
    return activation_freq / mean_freq


def get_feat_mask_adaptive(features, mask_rate, weights):
    """自适应掩码：高频特征更容易被保留"""
    N, D = features.shape
    col_prob = (mask_rate * weights).clamp(0.0, 1.0)
    return (torch.rand(N, D, device=weights.device) < col_prob.unsqueeze(0)).float()


def apply_mask_token(features, mask, mask_token):
    """用可学习 mask token 替换被遮蔽位置"""
    return features * (1 - mask) + mask * mask_token.expand(features.shape[0], -1)


def loss_function(preds, labels):
    """加权二元交叉熵（用于图重构损失）"""
    pos_weight = (labels.shape[0] ** 2 - labels.sum()) / labels.sum()
    norm = labels.shape[0] ** 2 / ((labels.shape[0] ** 2 - labels.sum()) * 2)
    return norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)


def sce_loss(x, y, alpha=3):
    """归一化余弦相似度幂损失（用于特征重建）"""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return (1 - (x * y).sum(dim=-1)).pow_(alpha).mean()


def compute_total_loss(vae_loss, topo_loss, mse_loss, loss_dis,
                       lambda_str, lambda_topo):
    return mse_loss + lambda_str * vae_loss + lambda_topo * (topo_loss + loss_dis)


# ==================== 主训练函数 ====================

def main(args):
    warnings.filterwarnings("ignore")
    os.makedirs(args['save_dir'], exist_ok=True)

    # ---------- 加载 Yelp 数据 ----------
    g, train_idx, val_idx, test_idx, ADJ_TOPO, ADJ = load_data_YELP()
    ADJ = [graph.to(args['device']) for graph in ADJ]
    topo_dim = ADJ_TOPO.shape[1]

    # ---------- 加载多个元路径嵌入并拼接 ----------
    z_pre1 = sp.load_npz(r"/home/hly/project/HGMAE-LBD/YELP/metapath2vec_blb.npz").A
    z_pre2 = sp.load_npz(r"/home/hly/project/HGMAE-LBD/YELP/metapath2vec_bub.npz").A
    z_pre3 = sp.load_npz(r"/home/hly/project/HGMAE-LBD/YELP/metapath2vec_bsb.npz").A

    z_pre = torch.FloatTensor(np.concatenate([z_pre1, z_pre2, z_pre3], axis=1)).to(args['device'])
    target = 'business'
    z_pre = z_pre[:g.num_nodes(target), :]
    zpre_dim = z_pre.shape[1]

    ADJ_TOPO = ADJ_TOPO.to(args['device'])
    featsdim = g.ndata["feature"][target].shape[1]
    x_true = g.ndata["feature"][target].to(args['device'])

    # ---------- 标签处理 ----------
    labels = g.ndata['label'][target].squeeze().to(args['device'])
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)
    numclass = int(labels.max().item()) + 1
    print(f"Number of classes: {numclass}")

    g = g.to(args['device'])
    print(f"Canonical etypes: {g.canonical_etypes}")

    in_feats = {ntype: g.ndata["feature"][ntype].shape[1] for ntype in g.ntypes}
    hidden_feats = 128
    out_feats = numclass
    num_layers = 2
    aggregator_type_dict = {ntype: 'mean' for ntype in g.ntypes}

    feat_weights = compute_activation_weights(g.ndata['feature'][target]).to(args['device'])
    topo_true = compute_topo_target(g, target, x_true, args['device'])

    mac_list = []
    mic_list = []
    nmi_list = []
    ari_list = []

    print(f'Start training, repeat = {args["repeat"]}\n')

    for cur_repeat in range(args['repeat']):
        set_random_seed(args['seed'] + cur_repeat)
        print(f'cur_repeat = {cur_repeat} {"=" * 60}')

        # ---------- 模型初始化 ----------
        model = HeteroGraphSAGE(
            g, in_feats, hidden_feats, out_feats, num_layers,
            aggregator_type_dict, featsdim, topo_dim,
            args['dataset'], float(args['alpha']), args['Init'],
            args['rank'], args['K'],
            attn_drop=args['attn_drop'],
            beta=args['beta'],
            zpre_dim=zpre_dim
        ).to(args['device'])

        save_path = os.path.join(args['save_dir'],
                                 f"checkpoint_{args['dataset']}_{cur_repeat}.pt")
        early_stopping = EarlyStopping(patience=args['patience'], verbose=True,
                                       save_path=save_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                     weight_decay=args['weight_decay'])

        for epoch in range(args['num_epochs']):
            mask_v1 = get_feat_mask_adaptive(
                g.ndata['feature'][target].to(args['device']),
                args['mask_rate'], feat_weights)
            features_v1 = apply_mask_token(
                g.ndata['feature'][target].to(args['device']),
                mask_v1, model.mask_token)

            w_dis_actual = args['w_dis'] * min(1.0, (epoch + 1) / max(args['warmup_epochs'], 1))

            # ---------- 训练 ----------
            model.train()
            optimizer.zero_grad()

            logits, h, x_pre, z_pre1, ADJ_P, loss_dis, topo_pre = model(
                g, ADJ_TOPO, z_pre, features_v1, target,
                args['tau'], w_dis_actual, args['kl2_weight'])

            vae_loss = sum(loss_function(ADJ_P, adj) for adj in ADJ) / len(ADJ)
            mse_loss = sce_loss(x_pre, x_true, alpha=args['sce_alpha'])
            topo_loss = sce_loss(topo_pre, topo_true, alpha=args['sce_alpha'])
            loss = compute_total_loss(vae_loss, topo_loss, mse_loss, loss_dis,
                                      args['lambda_str'], args['lambda_topo'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['grad_clip'])
            optimizer.step()

            # ---------- 验证 ----------
            model.eval()
            with torch.no_grad():
                fc = g.ndata['feature'][target].to(args['device'])
                _, hv, xv, _, Av, ldv, tpv = model(
                    g, ADJ_TOPO, z_pre, fc, target,
                    args['tau'], w_dis_actual, args['kl2_weight'])
                vae_v = sum(loss_function(Av, adj) for adj in ADJ) / len(ADJ)
                mse_v = sce_loss(xv, x_true, alpha=args['sce_alpha'])
                # 【Bug 修复点】：验证集原对比项错写为 x_true，现修正为 topo_true
                top_v = sce_loss(tpv, topo_true, alpha=args['sce_alpha'])
                val_loss = compute_total_loss(vae_v, top_v, mse_v, ldv,
                                              args['lambda_str'], args['lambda_topo'])

            print(
                f'Epoch {epoch + 1:4d} | Train {loss.item():.4f} | Val {val_loss.item():.4f} | loss_dis {loss_dis.item():.6f}')

            if torch.isnan(loss) or torch.isnan(val_loss):
                print(f'  [Warning] NaN at epoch {epoch + 1}, skipping.')
                continue
            early_stopping(val_loss.item(), model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # ---------- 测试 ----------
        print('\nTesting...')
        model.load_state_dict(torch.load(save_path))
        model.eval()
        with torch.no_grad():
            fc = g.ndata['feature'][target].to(args['device'])
            _, h_test, _, _, _, _, _ = model(
                g, ADJ_TOPO, z_pre, fc, target,
                args['tau'], args['w_dis'], args['kl2_weight'])

        mac, mic, nmi, ari = evaluate_results_nc(
            h_test[test_idx].detach().cpu().numpy(),
            labels[test_idx].cpu().numpy(), numclass)

        mac_list.append(mac)
        mic_list.append(mic)
        nmi_list.append(nmi)
        ari_list.append(ari)

    mac_arr = np.array(mac_list)
    mic_arr = np.array(mic_list)
    nmi_arr = np.array(nmi_list)
    ari_arr = np.array(ari_list)

    mac_mean, mac_std = np.mean(mac_arr, axis=0), np.std(mac_arr, axis=0)
    mic_mean, mic_std = np.mean(mic_arr, axis=0), np.std(mic_arr, axis=0)
    nmi_mean, nmi_std = np.mean(nmi_arr), np.std(nmi_arr)
    ari_mean, ari_std = np.mean(ari_arr), np.std(ari_arr)


    print(f'\n--- Summary of {args["repeat"]} runs (Mean ± Std) ---')
    print('Macro-F1:')
    for mean_v, std_v in zip(mac_mean, mac_std):
        print(f'    {mean_v:.6f} ± {std_v:.6f}')

    print('Micro-F1:')
    for mean_v, std_v in zip(mic_mean, mic_std):
        print(f'    {mean_v:.6f} ± {std_v:.6f}')

    print(f'NMI: {nmi_mean:.6f} ± {nmi_std:.6f}')
    print(f'ARI: {ari_mean:.6f} ± {ari_std:.6f}')
    print('All finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALRE Enhanced HeteroGraphSAGE on Yelp')

    # ---------- 数据与路径 ----------
    parser.add_argument('--dataset', default='YELP')
    parser.add_argument('--save_dir', default='./checkpoints')

    # ---------- 训练超参数 ----------
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # ---------- 数据增强 ----------
    parser.add_argument('--mask_rate', type=float, default=0.1)

    # ---------- 损失权重 ----------
    parser.add_argument('--lambda_str', type=float, default=0.3)
    parser.add_argument('--lambda_topo', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--w_dis', type=float, default=0.05)
    parser.add_argument('--kl2_weight', type=float, default=1.0)
    parser.add_argument('--sce_alpha', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.5)

    # ---------- 模型结构 ----------
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--Init', type=str, default='Mine_PPR')
    parser.add_argument('--rank', type=int, default=3)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--attn_drop', type=float, default=0.0)

    # ---------- 设备 ----------
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args().__dict__
    print(args)
    main(args)

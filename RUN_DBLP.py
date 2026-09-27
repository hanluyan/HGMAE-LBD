import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from load_data import load_data_DBLP
from model import HeteroGraphSAGE          # 使用 ALRE 增强模型
from tools import evaluate_results_nc, EarlyStopping


# ==================== 工具函数 ====================
def compute_topo_target(g, target, x_true, device):
    N = g.num_nodes(target)
    feat_dim = x_true.shape[1]
    agg = torch.zeros(N, feat_dim, device=device)
    cnt = torch.zeros(N, 1, device=device)

    for stype, etype, dtype in g.canonical_etypes:
        if dtype != target:
            continue
        src, dst = g[stype, etype, dtype].edges()
        src = src.to(device).long()   # ← 加 .long()
        dst = dst.to(device).long()   # ← 加 .long()

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
    activation_freq = (features != 0).float().mean(dim=0)
    mean_freq = activation_freq.mean().clamp(min=1e-8)
    return activation_freq / mean_freq



def get_feat_mask_adaptive(features, mask_rate, weights):
    N, D = features.shape
    col_prob = (mask_rate * weights).clamp(0.0, 1.0)
    return (torch.rand(N, D, device=weights.device) < col_prob.unsqueeze(0)).float()


def apply_mask_token(features, mask, mask_token):
    return features * (1 - mask) + mask * mask_token.expand(features.shape[0], -1)


def loss_function(preds, labels):
    # 修复：pos_weight 应为标量
    pos_weight = (labels.shape[0] ** 2 - labels.sum()) / labels.sum()
    norm = labels.shape[0] ** 2 / ((labels.shape[0] ** 2 - labels.sum()) * 2)
    return norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return (1 - (x * y).sum(dim=-1)).pow_(alpha).mean()


def compute_total_loss(vae_loss, topo_loss, mse_loss, loss_dis,
                       lambda_str, lambda_topo):
    return mse_loss + lambda_str * vae_loss + lambda_topo * (topo_loss + loss_dis)


# ==================== 训练主函数 ====================

def main(args):
    warnings.filterwarnings("ignore")
    os.makedirs(args['save_dir'], exist_ok=True)

    # ===== 数据加载（DBLP） =====
    g, train_idx, val_idx, test_idx, ADJ_TOPO, ADJ = load_data_DBLP()
    ADJ = [graph.to(args['device']) for graph in ADJ]
    topo_dim = ADJ_TOPO.shape[1]

    z_pre = torch.FloatTensor(np.load(args['pretrain_emb_path'])).to(args['device'])
    target = 'author'                # DBLP 目标节点类型
    z_pre = z_pre[:g.num_nodes(target), :]
    zpre_dim = z_pre.shape[1]        # 获取预训练嵌入维度

    ADJ_TOPO = ADJ_TOPO.to(args['device'])
    featsdim = g.ndata["feature"][target].shape[1]
    x_true = g.ndata["feature"][target].to(args['device'])

    labels = g.ndata['label'][target].squeeze().to(args['device'])
    numclass = int(labels.max().item()) + 1 if labels.dim() == 1 else labels.shape[1]

    print(f"Number of classes: {numclass}")
    g = g.to(args['device'])
    print(f"Canonical etypes: {g.canonical_etypes}")

    in_feats = {ntype: g.ndata["feature"][ntype].shape[1] for ntype in g.ntypes}
    hidden_feats = 64
    out_feats = numclass
    num_layers = 2
    aggregator_type_dict = {ntype: 'mean' for ntype in g.ntypes}

    svm_macro_avg = np.zeros((7,), dtype=np.float64)
    svm_micro_avg = np.zeros((7,), dtype=np.float64)
    nmi_avg = ari_avg = 0.0

    print(f'Start training, repeat = {args["repeat"]}\n')
    feat_weights = compute_activation_weights(
        g.ndata['feature'][target]
    ).to(args['device'])
    topo_true = compute_topo_target(g, target, x_true, args['device'])
    for cur_repeat in range(args['repeat']):
        set_random_seed(args['seed'] + cur_repeat)
        print(f'cur_repeat = {cur_repeat} {"=" * 60}')

        # ===== 模型初始化（ALRE 增强版）=====
        model = HeteroGraphSAGE(
            g, in_feats, hidden_feats, out_feats, num_layers,
            aggregator_type_dict, featsdim, topo_dim,
            args['dataset'], float(args['alpha']), args['Init'],
            args['rank'], args['K'],
            attn_drop=args['attn_drop'],
            beta=args['beta'],
            zpre_dim=zpre_dim                     # 动态适配预训练嵌入维度
        ).to(args['device'])

        save_path = os.path.join(
            args['save_dir'],
            f"checkpoint_{args['dataset']}_alre_{cur_repeat}.pt"
        )
        early_stopping = EarlyStopping(patience=args['patience'], verbose=True,
                                       save_path=save_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                     weight_decay=args['weight_decay'])

        for epoch in range(args['num_epochs']):
            # 自适应掩码
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
                top_v = sce_loss(tpv, topo_true, alpha=args['sce_alpha'])
                val_loss = compute_total_loss(vae_v, top_v, mse_v, ldv,
                                              args['lambda_str'], args['lambda_topo'])

            # print(f'Epoch {epoch+1:4d} | Train {loss.item():.4f} | Val {val_loss.item():.4f}')
            print(f'Epoch {epoch + 1:4d} | '
                  f'Train {loss.item():.4f} | '
                  f'VAE {vae_loss.item():.4f} | '
                  f'Topo {topo_loss.item():.4f} | '
                  f'MSE {mse_loss.item():.4f} | '
                  f'Dis {loss_dis.item():.4f} | '
                  f'Val {val_loss.item():.4f}')

            if torch.isnan(loss) or torch.isnan(val_loss):
                print(f'  [Warning] NaN at epoch {epoch+1}, skipping.')
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
        svm_macro_avg += mac; svm_micro_avg += mic
        nmi_avg += nmi; ari_avg += ari

    svm_macro_avg /= args['repeat']; svm_micro_avg /= args['repeat']
    nmi_avg /= args['repeat']; ari_avg /= args['repeat']

    print(f'\n--- Average of {args["repeat"]} runs ---')
    print('Macro-F1:'); [print(f'    {v:.6f}') for v in svm_macro_avg]
    print('Micro-F1:'); [print(f'    {v:.6f}') for v in svm_micro_avg]
    print(f'NMI: {nmi_avg:.6f}')
    print(f'ARI: {ari_avg:.6f}')
    print('All finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALRE Enhanced HeteroGraphSAGE on DBLP')

    # 数据与路径
    parser.add_argument('--dataset', default='DBLP')
    parser.add_argument('--pretrain_emb_path', default='DBLP/metapath2vec_emb.npy')
    parser.add_argument('--save_dir', default='./checkpoints')

    # 训练超参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0.0003)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # 数据增强
    parser.add_argument('--mask_rate', type=float, default=0.2)

    # 损失权重（DBLP 默认值）
    parser.add_argument('--lambda_str', type=float, default=0.9)
    parser.add_argument('--lambda_topo', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--w_dis', type=float, default=0.15)
    parser.add_argument('--kl2_weight', type=float, default=1.0)
    parser.add_argument('--sce_alpha', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.0)

    # 模型结构
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--Init', type=str, default='PPR')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--attn_drop', type=float, default=0.0)

    # 设备
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args().__dict__
    print(args)
    main(args)

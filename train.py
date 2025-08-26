import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from torch import amp
torch.set_float32_matmul_precision("high")

EOS_TOKEN_IDX = 17

def create_soft_label_matrix(labels: torch.Tensor,
                             num_classes: int,
                             n_s: int = 5,
                             w_m: float = 1.0,
                             w_s: float = 0.1,
                             return_reshaped: bool = False):
    """
    Vectorized soft label builder.

    labels: Long tensor of class ids. Shape can be arbitrary (e.g. [B, T-1] or [N]).
    num_classes: int, number of classes (e.g., C_time).
    n_s: radius for side neighbors.
    w_m: weight for the main (true) class.
    w_s: weight for each side neighbor.
    return_reshaped: if True, returns [..., num_classes] matching labels' original shape.
                     if False, returns [N, num_classes] flattened (useful for matmul with logits).

    Returns:
      soft: Tensor of shape [N, C] if return_reshaped=False,
            else shape [..., C] (same leading dims as labels).
    """
    # ---- sanitize inputs ----
    if labels.dtype != torch.long:
        labels = labels.long()
    device = labels.device
    orig_shape = labels.shape
    labels_flat = labels.reshape(-1)                  # [N]
    N = labels_flat.numel()
    C = int(num_classes)

    # offsets and weights
    offsets = torch.arange(-n_s, n_s + 1, device=device)             # [K], K=2*n_s+1
    K = offsets.numel()
    weights = torch.full((K,), w_s, device=device)
    weights[n_s] = w_m                                              # center

    # indices per sample for scatter_add
    idx = (labels_flat.unsqueeze(1) + offsets.unsqueeze(0))          # [N, K]
    idx.clamp_(0, C - 1)

    # build soft matrix
    soft = torch.zeros(N, C, device=device, dtype=torch.float32)     # [N, C]
    soft.scatter_add_(1, idx, weights.expand(N, K))                  # add neighbor weights

    # normalize rows to 1 (avoid divide-by-zero)
    row_sum = soft.sum(dim=1, keepdim=True).clamp_min_(1e-9)
    soft = soft / row_sum

    if return_reshaped:
        soft = soft.view(*orig_shape, C)                             # [..., C]
    return soft


# def create_soft_label_matrix(original_labels, num_classes=96, n_s=5, w_m=1.0, w_s=0.1):
#     """
#     original_labels: [N] LongTensor of class indices
#     returns: [N, num_classes] soft label distributions
#     """
#     B = original_labels.size(0)
#     soft_labels = torch.zeros((B, num_classes), device=original_labels.device)

#     for i in range(B):
#         true_idx = int(original_labels[i].item())
#         soft_labels[i, true_idx] = w_m
#         for offset in range(1, n_s + 1):
#             if true_idx - offset >= 0:
#                 soft_labels[i, true_idx - offset] = w_s
#             if true_idx + offset < num_classes:
#                 soft_labels[i, true_idx + offset] = w_s

#     # Normalize row to sum to 1
#     soft_labels = soft_labels / torch.clamp(soft_labels.sum(dim=1, keepdim=True), min=1e-9)
#     return soft_labels

def train_model(model,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                num_epochs: int = 20,
                lr: float = 1e-4,
                weight_decay: float = 1e-5,
                scheduler_step: int = 5,
                scheduler_gamma: float = 0.5,
                early_stopping_patience: int = 3,
                loss_weights: dict = None,
                writer=None
                ):
    """
    Batch-first training loop (no permutes)
    """
    scaler = amp.GradScaler(enabled=(device.type=="cuda"))
    if loss_weights is None:
        loss_weights = {'L_CE': 1.0, 'L_s': 1.0, 'L_e': 1.0, 'L_o': 1.0, 'L_seq': 1.0}
    
    pad_idx = 0
    # 손실 함수 정의
    type_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Early stopping 변수
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            # --- inputs ---
            activity_chain      = batch['activity_chain'].to(device, non_blocking=True).long()
            target_features     = batch['target_features'].to(device, non_blocking=True).long()
            household_members   = batch['household_members'].to(device, non_blocking=True).long()
            household_mask      = batch['household_mask'].to(device, non_blocking=True).bool()

            # labels per token
            type_labels_seq     = activity_chain[..., 0].long() # [B, T]
            start_labels_seq    = activity_chain[..., 1].long() # [B, T]
            end_labels_seq      = activity_chain[..., 2].long() # [B, T]

            # teacher-forcing target (shifted right)
            tgt_activity = activity_chain[:, :-1, :3].long()

            # 순전파
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
                type_logits, start_logits, end_logits = model(
                    src_activity=activity_chain,            # [B, T, 5]
                    person_info=target_features,            # [B, 1, Dp]
                    household_info=household_members,       # [B, H, Dh]
                    tgt_activity=tgt_activity,              # [B, T-1, 3]
                    household_padding_mask=household_mask   # [B, H] bool
                    )   # each: [B, T-1, C]
                
                B, Tm1, C_type = type_logits.shape
                C_time = start_logits.shape[-1]

                # valid token mask for targets (based on stored length at time 0, feature index 3)
                T_max = activity_chain.size(1)
                tgt_len = (activity_chain[:, 0, 3].long() + 1).clamp(max=T_max)     # [B]
                idx = torch.arange(T_max - 1, device=device).unsqueeze(0)           # [1, T-1]
                tgt_mask = (idx < tgt_len.unsqueeze(1)).to(torch.bool)              # [B, T-1] bool
                mask_flat = tgt_mask.reshape(-1).float()            # [(B*(T-1))]

                # 1) CrossEntropy for Activity Type
                type_logits_flat  = type_logits.view(-1, C_type)
                type_targets_flat = type_labels_seq[:, 1:].reshape(-1)  # 디코더 예측 스텝에 맞춰 시프트
                ce_losses = type_criterion(type_logits_flat, type_targets_flat)

                # 2) Soft-label Losses for Start/End
                P_start = F.log_softmax(start_logits.reshape(-1, C_time),   dim=1)
                P_end   = F.log_softmax(end_logits.reshape(-1, C_time),     dim=1)
                SLM_start   = create_soft_label_matrix(start_labels_seq[:, 1:].reshape(-1).to(device), num_classes=C_time)
                SLM_end     = create_soft_label_matrix(end_labels_seq[:, 1:].reshape(-1).to(device), num_classes=C_time)
                soft_losses_start   = -(SLM_start   * P_start).sum(dim=1)   # [(T-1)*B]
                soft_losses_end     = -(SLM_end     * P_end  ).sum(dim=1)

                # 3) Temporal Order & Overlap Penalties
                preds_start = start_logits.argmax(dim=-1)       # [B, T-1]
                preds_end   = end_logits.argmax(dim=-1)         # [B, T-1]

                # L_seq: penalizes start_time > end_time
                dur_violation = torch.relu(preds_start.float() - preds_end.float())
                L_seq = dur_violation.masked_select(tgt_mask).mean()

                # L_o: penalizes overlap between consecutive activities
                if Tm1 > 1:
                    overlap = torch.relu(preds_end[:, :-1].float() - preds_start[:, 1:].float())    # [B, T-2]
                    valid_pair_mask = tgt_mask[:, :-1] & tgt_mask[:, 1:]                            # [B, T-2]
                    L_o = overlap.masked_select(valid_pair_mask).mean()
                else:
                    L_o = torch.tensor(0.0, device=device)
                
                L_CE    = (ce_losses            * mask_flat).sum() / (mask_flat.sum() + 1e-9)
                L_s     = (soft_losses_start    * mask_flat).sum() / (mask_flat.sum() + 1e-9)
                L_e     = (soft_losses_end      * mask_flat).sum() / (mask_flat.sum() + 1e-9)

                loss = (loss_weights['L_CE'] * L_CE +
                        loss_weights['L_s'] * L_s +
                        loss_weights['L_e'] * L_e +
                        loss_weights['L_o'] * L_o +
                        loss_weights['L_seq'] * L_seq)

            # 역전파 및 최적화
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach()) * B

        epoch_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()
        if writer is not None:
            writer.add_scalar("Loss/train", epoch_train_loss, epoch)

        # ------------------------Validation------------------------
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # --- inputs ---
                activity_chain      = batch['activity_chain'].to(device).long()
                target_features     = batch['target_features'].to(device).long()
                household_members   = batch['household_members'].to(device).long()
                household_mask      = batch['household_mask'].to(device).bool()

                # labels per token
                type_labels_seq     = activity_chain[..., 0].long() # [B, T]
                start_labels_seq    = activity_chain[..., 1].long() # [B, T]
                end_labels_seq      = activity_chain[..., 2].long() # [B, T]
                tgt_activity        = activity_chain[:, :-1, :3].long()

                # 순전파
                with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
                    type_logits, start_logits, end_logits = model(
                        src_activity=activity_chain,            # [B, T, 5]
                        person_info=target_features,            # [B, 1, Dp]
                        household_info=household_members,       # [B, H, Dh]
                        tgt_activity=tgt_activity,              # [B, T-1, 3]
                        household_padding_mask=household_mask   # [B, H] bool
                        )   # each: [B, T-1, C]
                    
                    B, Tm1, C_type = type_logits.shape
                    C_time = start_logits.shape[-1]
                    T_max = activity_chain.size(1)
                    tgt_len = (activity_chain[:, 0, 3].long() + 1).clamp(max=T_max)
                    idx = torch.arange(T_max - 1, device=device).unsqueeze(0)
                    tgt_mask = (idx < tgt_len.unsqueeze(1)).to(torch.bool)
                    mask_flat = tgt_mask.reshape(-1).float()

                    # 1) CrossEntropy for Activity Type
                    type_logits_flat  = type_logits.reshape(-1, C_type)
                    type_targets_flat = type_labels_seq[:, 1:].reshape(-1)  # 디코더 예측 스텝에 맞춰 시프트
                    ce_losses = type_criterion(type_logits_flat, type_targets_flat)

                    # 2) Soft-label Losses for Start/End
                    P_start = F.log_softmax(start_logits.reshape(-1, C_time), dim=1)
                    P_end   = F.log_softmax(end_logits.reshape(-1, C_time),   dim=1)
                    SLM_start   = create_soft_label_matrix(start_labels_seq[:, 1:].reshape(-1).to(device),  num_classes=C_time)
                    SLM_end     = create_soft_label_matrix(end_labels_seq[:, 1:].reshape(-1).to(device),    num_classes=C_time)
                    soft_losses_start   = -(SLM_start   * P_start).sum(dim=1)   # [(T-1)*B]
                    soft_losses_end     = -(SLM_end     * P_end  ).sum(dim=1)

                    # 3) Temporal Order & Overlap Penalties
                    preds_start = start_logits.argmax(dim=-1)    # [T-1, B]
                    preds_end   = end_logits.argmax(dim=-1)
                    dur_violation = torch.relu(preds_start.float() - preds_end.float())
                    L_seq = dur_violation.masked_select(tgt_mask).mean()

                    if Tm1 > 1:
                        overlap = torch.relu(preds_end[:, :-1].float() - preds_start[:, 1:].float())    # [B, T-2]
                        valid_pair_mask = tgt_mask[:, :-1] & tgt_mask[:, 1:]                            # [B, T-2]
                        L_o = overlap.masked_select(valid_pair_mask).mean()
                    else:
                        L_o = torch.tensor(0.0, device=device)

                    
                    L_CE    = (ce_losses            * mask_flat).sum() / (mask_flat.sum() + 1e-9)
                    L_s     = (soft_losses_start    * mask_flat).sum() / (mask_flat.sum() + 1e-9)
                    L_e     = (soft_losses_end      * mask_flat).sum() / (mask_flat.sum() + 1e-9)

                    loss = (loss_weights['L_CE'] * L_CE +
                            loss_weights['L_s'] * L_s +
                            loss_weights['L_e'] * L_e +
                            loss_weights['L_o'] * L_o +
                            loss_weights['L_seq'] * L_seq)

                val_running_loss += float(loss.detach()) * activity_chain.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        if writer is not None:
            writer.add_scalar("Loss/val", epoch_val_loss, epoch)

        # Early stopping 체크
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pt')  # 최적 모델 저장
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"No improvement for {early_stopping_patience} epochs. Stopping training.")
                break

    # 최적 모델 로드
    model.load_state_dict(torch.load('best_model.pt'))
    return model

def predict_sequence(model, src_activity, person_info, household_info, household_padding_mask, device, max_len=16):
    """
    Autoregressive inference (batch-first, B may be 1).
    src_activity: [B, T, 5], person_info: [B,1,Dp], household_info: [B,H,Dh], mask: [B,H]
    returns: [B, L, 3]
    """
    model.eval()

    with torch.no_grad():
        # 1. Encode the source inputs once
        memory = model.encoder(src_activity, person_info, household_info, household_padding_mask)   # [B, S, h2]

        # The memory padding mask needs to be constructed just like in the training forward pass
        B = src_activity.size(0)
        device = src_activity.device
        person_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        h_mask = household_padding_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(B, -1)
        final_sep_and_activity_mask = torch.zeros((B, 1 + src_activity.size(1)), dtype=torch.bool, device=device)
        full_padding_mask = torch.cat([person_mask, h_mask, final_sep_and_activity_mask], dim=1) # [B, S]

        # 2. Start with a <SOS> token. Let's assume the <SOS> token is represented by index 0 for all 3 features.
        # Shape: [sequence_len, batch_size, num_features] -> [1, 1, 3]
        dec_inp = torch.zeros((B, 1, 3), dtype=torch.long, device=device)

        # 3. Autoregressive loop
        generated = []
        for _ in range(max_len):
            # Get predictions from the decoder
            type_logits, start_logits, end_logits = model.decoder(
                dec_inp, memory, memory_key_padding_mask=full_padding_mask
            )   # [B, cur_len, C]

            # Focus only on the last token in the sequence for the next prediction
            # Shape of logits: [batch_size, current_seq_len, vocab_size]
            last_type = type_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)       # [B, 1]
            last_start = start_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)     # [B, 1]
            last_end = end_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)         # [B, 1]
            next_token = torch.stack([last_type, last_start, last_end], dim=-1) # [B, 1, 3]

            # if last_type.item() == EOS_TOKEN_IDX:
            #     break

            dec_inp = torch.cat([dec_inp, next_token], dim=1) # append along time
            generated.append(next_token)

    return torch.cat(generated, dim=1)  # [B, L, 3]

if __name__ == "__main__":
    from model.model import FullActivityTransformer
    from dataset.dataset import ActivityDataset
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
    PERSONS_PATH = "C:/Users/user/PTV_Intern/src/DeepAM/dataset/persons_info.csv"
    CHAIN_PATH   = "C:/Users/user/PTV_Intern/src/DeepAM/dataset/activity_chain.npy"
    IDS_PATH     = "C:/Users/user/PTV_Intern/src/DeepAM/dataset/person_ids.csv"
    W_PATH       = "C:/Users/user/PTV_Intern/src/DeepAM/dataset/chain_weights.npy"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device selected: ", device)
    
    activity_chains = np.load(CHAIN_PATH)               # expected [N, T, F]
    all_person_df   = pd.read_csv(PERSONS_PATH, index_col=0)
    person_ids_df   = pd.read_csv(IDS_PATH, index_col=0)
    full_dataset    = ActivityDataset(activity_chains, all_person_df, person_ids_df)
    
    # Get vocab sizes
    PERSON_VOCAB = [len(all_person_df[c].unique()) +1 for c in all_person_df.columns if c not in ['HOUSEID', 'PERSONID']]
    HOUSEHOLD_VOACB = []
    for col in full_dataset.hh_members_features:
        HOUSEHOLD_VOACB.append(len(all_person_df[col].unique())+1)

    ACTIVITY_CHAIN_VOCAB = [activity_chains[:,:,i].max()+1 for i in range(activity_chains.shape[-1])]   # [type=15 + 2 (SOS, EOS), start=96, end=96, len=14, duration=96] + 1 for every vocab for padding
    TGT_ACTIVITY_CHAIN_VOCAB = ACTIVITY_CHAIN_VOCAB[:3]

    total_size = len(full_dataset)
    indices = np.arange(total_size)
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(indices, test_size=0.5, random_state=42)

    train_set   = Subset(full_dataset, train_idx)
    val_set     = Subset(full_dataset, val_idx)
    test_set    = Subset(full_dataset, test_idx)
    
    full_weights  = np.load(W_PATH)
    train_weights = full_weights[train_idx]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    num_workers = max(4, os.cpu_count() //2)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)
    
    model = FullActivityTransformer(h2=128,
                                nhead=8,
                                enc_layers=2,
                                dec_layers=2,
                                d_hid = 2,
                                src_act_vocab=ACTIVITY_CHAIN_VOCAB,
                                src_act_embed=6,
                                person_vocab=PERSON_VOCAB,
                                person_embed=6,
                                household_vocab=HOUSEHOLD_VOACB,
                                household_embed=6,
                                tgt_act_vocab=TGT_ACTIVITY_CHAIN_VOCAB,
                                tgt_act_embed=6,
                                dropout=0.1
    ).to(device)

    # Define loss weights
    loss_weights = {
        'L_CE': 1.0, 
        'L_s': 1.0, 
        'L_e': 1.0, 
        'L_o': 0.5, 
        'L_seq': 0.5
    }

    train_model(model=model,
                train_loader=train_loader,
                val_loader = val_loader,
                device=device,
                num_epochs=150,
                lr=0.005,
                weight_decay=0.00001,
                scheduler_step=5,
                scheduler_gamma=0.5,
                early_stopping_patience=5,
                loss_weights=loss_weights
                )

    

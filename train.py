import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

def create_soft_label_matrix(original_labels, num_classes=96, n_s=5, w_m=1.0, w_s=0.1):
    B = original_labels.size(0)
    soft_labels = torch.zeros((B, num_classes), device=original_labels.device)

    for i in range(B):
        true_idx = original_labels[i].item()
        soft_labels[i, true_idx] = w_m
        for offset in range(1, n_s + 1):
            if true_idx - offset >= 0:
                soft_labels[i, true_idx - offset] = w_s
            if true_idx + offset < num_classes:
                soft_labels[i, true_idx + offset] = w_s

    # Normalize row to sum to 1
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)

    return soft_labels

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
                loss_weights: dict = None
                ):
    """
    모델 학습 루프와 손실 함수 정의
    """
    if loss_weights is None:
        loss_weights = {'L_CE': 1.0, 'L_s': 1.0, 'L_e': 1.0, 'L_o': 1.0, 'L_seq': 1.0}

    # 손실 함수 정의
    type_criterion = nn.CrossEntropyLoss(reduction='none')

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
            # 입력 및 타겟 분리
            activity_chain = batch['activity_chain'].to(device)
            target_features = batch['target_features'].to(device)
            household_members = batch['household_members'].to(device)
            household_mask = batch['household_mask'].to(device)

            # Extract labels
            type_labels_seq = activity_chain[:, :, 0].long()
            start_labels_seq = activity_chain[:, :, 1].long()
            end_labels_seq = activity_chain[:, :, 2].long()

            # Teacher Forcing Input
            tgt_activity = activity_chain[:, :-1, :].permute(1, 0, 2).long().to(device)

            # 순전파
            optimizer.zero_grad()
            type_logits, start_logits, end_logits = model(activity_chain, target_features, household_members, tgt_activity, household_mask)
            
            T, B, C_type = type_logits.shape
            C_time = start_logits.shape[-1]

            # Masking
            tgt_len = activity_chain[:, 0, 3].long()
            T_max = activity_chain.size(1)

            idx = torch.arange(T_max - 1, device=device).unsqueeze(0)
            tgt_mask = idx < tgt_len

            mask_flat = tgt_mask.transpose(0, 1).reshape(-1).float()

            # 1) CrossEntropy for Activity Type
            type_logits_flat  = type_logits.view(-1, C_type)
            type_targets_flat = type_labels_seq[:, 1:].reshape(-1)  # 디코더 예측 스텝에 맞춰 시프트
            ce_losses = type_criterion(type_logits_flat, type_targets_flat, reductino='none')

            # 2) Soft-label Losses for Start/End
            P_start = F.log_softmax(start_logits.view(-1, C_time), dim=1)
            P_end = F.log_softmax(end_logits.view(-1, C_time), dim=1)

            SLM_start = create_soft_label_matrix(start_labels_seq[:, 1:].reshape(-1))
            SLM_end = create_soft_label_matrix(end_labels_seq[:, 1:].reshape(-1))

            soft_losses_start = -(SLM_start * P_start).sum(dim=1)   # [(T-1)*B]
            soft_losses_end = -(SLM_end * P_end).sum(dim=1)

            # 3) Temporal Order & Overlap Penalties
            preds_start = start_logits.argmax(dim=2)    # [T-1, B]
            preds_end   = end_logits.argmax(dim=2)

            # L_seq: penalizes start_time > end_time
            L_seq = torch.relu(preds_start.float() - preds_end.float()).masked_select(tgt_mask).mean()

            # L_o: penalizes overlap between consecutive activities
            L_o = torch.relu(preds_end[:-1,:,:].float() - preds_start[1:,:,:].float()).masked_select(tgt_mask[:-1]).mean()
            
            L_CE    = (ce_losses            * mask_flat).sum() / mask_flat.sum()
            L_s     = (soft_losses_start    * mask_flat).sum() / mask_flat.sum()
            L_e     = (soft_losses_end      * mask_flat).sum() / mask_flat.sum()

            loss = (loss_weights['L_CE'] * L_CE +
                    loss_weights['L_s'] * L_s +
                    loss_weights['L_e'] * L_e +
                    loss_weights['L_o'] * L_o +
                    loss_weights['L_seq'] * L_seq)

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * activity_chain.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # 검증
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                activity_chain = batch['activity_chain'].to(device)
                target_features = batch['target_features'].to(device)
                household_members = batch['household_members'].to(device)
                household_mask = batch['household_mask'].to(device)

                # Extract labels
                type_labels_seq = activity_chain[:, :, 0].long()
                start_labels_seq = activity_chain[:, :, 1].long()
                end_labels_seq = activity_chain[:, :, 2].long()

                # Teacher Forcing Input
                tgt_activity = activity_chain[:, :-1, :].permute(1, 0, 2).long().to(device)

                type_logits, start_logits, end_logits = model(activity_chain, target_features, household_members, tgt_activity, household_mask)
                
                T, B, C_type = type_logits.shape

                # 1) CrossEntropy for Activity Type
                type_logits_flat  = type_logits.view(T * B, C_type)
                type_targets_flat = type_labels_seq[:, 1:].reshape(-1)  # 디코더 예측 스텝에 맞춰 시프트
                L_CE = type_criterion(type_logits_flat, type_targets_flat)

                # 2) Soft-label Losses for Start/End
                P_start = F.log_softmax(start_logits.view(T * B, -1), dim=1)
                P_end = F.log_softmax(end_logits.view(T * B, -1), dim=1)

                SLM_start = create_soft_label_matrix(start_labels_seq[:, 1:].reshape(-1))
                SLM_end = create_soft_label_matrix(end_labels_seq[:, 1:].reshape(-1))
                
                L_s = -(SLM_start * P_start).sum(dim=1).mean()
                L_e = -(SLM_end * P_end).sum(dim=1).mean()
                
                # 3) Temporal Order & Overlap Penalties
                preds_start = start_logits.argmax(dim=2)    # [T-1, B]
                preds_end   = end_logits.argmax(dim=2)

                # L_seq: penalizes start_time > end_time
                L_seq = torch.relu(preds_start.float() - preds_end.float()).mean()

                # L_o: penalizes overlap between consecutive activities
                L_o = torch.relu(preds_end[:-1,:,:].float() - preds_start[1:,:,:].float()).mean()
                
                loss = (loss_weights['L_CE'] * L_CE +
                        loss_weights['L_s'] * L_s +
                        loss_weights['L_e'] * L_e +
                        loss_weights['L_o'] * L_o +
                        loss_weights['L_seq'] * L_seq)
                
                val_running_loss += loss.item() * activity_chain.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

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


if __name__ == "__main__":
    # 예시: 데이터 로더, 모델, 디바이스 설정 후 학습 호출
    from DeepAM.model.model import FullActivityTransformer
    from dataset.dataset import ActivityDataset
    PERSONS_PATH = "C:/Users/user/PTV_Intern/src/DeepAM/dataset/persons_info.csv"
    CHAIN_PATH = "C:/Users/user/PTV_Intern/src/DeepAM/dataset/activity_chain.npy"
    IDS_PATH = "C:/Users/user/PTV_Intern/src/DeepAM/dataset/person_ids.csv"
    W_PATH = "C:/Users/user/PTV_Intern/src/DeepAM/dataset/chain_weights.npy"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    

    
    activity_chains = np.load(CHAIN_PATH)
    all_person_df = pd.read_csv(PERSONS_PATH)
    person_ids_df = pd.read_csv(IDS_PATH)
    train_dataset = ActivityDataset(activity_chains, all_person_df, person_ids_df)
    
    
    weights = np.load(W_PATH)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    

    # Define loss weights
    loss_weights = {
        'L_CE': 1.0, 
        'L_s': 1.0, 
        'L_e': 1.0, 
        'L_o': 0.5, 
        'L_seq': 0.5
    }

    

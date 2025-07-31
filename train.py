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
    type_criterion = nn.CrossEntropyLoss()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Early stopping 변수
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            # 입력 및 타겟 분리
            input_ids = batch['input_ids'].to(device)           # [B, S]
            attention_mask = batch['attention_mask'].to(device) # [B, S]
            type_labels = batch['type_labels'].to(device)       # [B]
            start_labels = batch['start_labels'].to(device)     # [B]
            end_labels = batch['end_labels'].to(device)         # [B]
            SLM_start = create_soft_label_matrix(start_labels)  # [B, C]
            SLM_end = create_soft_label_matrix(end_labels)      # [B, C]

            optimizer.zero_grad()
            # 순전파
            type_logits, start_logits, end_logits = model(input_ids, attention_mask)
            P_start = F.log_softmax(start_logits, dim=1)        # [B, C]
            P_end = F.log_softmax(end_logits, dim=1)            # [B, C]

            # 손실 계산
            L_CE = type_criterion(type_logits, type_labels)
            L_s = -(SLM_start * P_start).sum(dim=1).mean()
            L_e = -(SLM_end * P_end).sum(dim=1).mean()
            
            # Get predicted indices for start and end times
            _, start_preds = torch.max(start_logits, 1)
            _, end_preds = torch.max(end_logits, 1)

            # L_seq: penalizes start_time > end_time
            L_seq = torch.mean(torch.relu(start_preds.float() - end_preds.float()))

            # L_o: penalizes overlap between consecutive activities
            if input_ids.size(0) > 1:
                end_preds_prev = end_preds[:-1]
                start_preds_curr = start_preds[1:]
                L_o = torch.mean(torch.relu(end_preds_prev.float() - start_preds_curr.float()))
            else:
                L_o = 0.0

            loss = (loss_weights['L_CE'] * L_CE +
                    loss_weights['L_s'] * L_s +
                    loss_weights['L_e'] * L_e +
                    loss_weights['L_o'] * L_o +
                    loss_weights['L_seq'] * L_seq)

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # 검증
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                type_labels = batch['type_labels'].to(device)
                start_labels = batch['start_labels'].to(device)
                end_labels = batch['end_labels'].to(device)
                SLM_start = create_soft_label_matrix(start_labels)
                SLM_end = create_soft_label_matrix(end_labels)

                type_logits, start_logits, end_logits = model(input_ids, attention_mask)
                P_start = F.log_softmax(start_logits, dim=1)
                P_end = F.log_softmax(end_logits, dim=1)

                L_CE = type_criterion(type_logits, type_labels)
                L_s = -(SLM_start * P_start).sum(dim=1).mean()
                L_e = -(SLM_end * P_end).sum(dim=1).mean()
                
                # Get predicted indices for start and end times
                _, start_preds = torch.max(start_logits, 1)
                _, end_preds = torch.max(end_logits, 1)

                # L_seq: penalizes start_time > end_time
                L_seq = torch.mean(torch.relu(start_preds.float() - end_preds.float()))

                # L_o: penalizes overlap between consecutive activities
                if input_ids.size(0) > 1:
                    end_preds_prev = end_preds[:-1]
                    start_preds_curr = start_preds[1:]
                    L_o = torch.mean(torch.relu(end_preds_prev.float() - start_preds_curr.float()))
                else:
                    L_o = 0.0
                
                loss = (loss_weights['L_CE'] * L_CE +
                        loss_weights['L_s'] * L_s +
                        loss_weights['L_e'] * L_e +
                        loss_weights['L_o'] * L_o +
                        loss_weights['L_seq'] * L_seq)
                val_running_loss += loss.item() * input_ids.size(0)

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
    # model = FullActivityTransformer(...).to(device) # Replace with your model initialization
    # train_dataset = ActivityDataset(split='train')
    # val_dataset = ActivityDataset(split='val')
    # sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    # train_loader = DataLoader(datset, batch_size=64, sampler=sampler)
    # val_loader = DataLoader(val_dataset, batch_size=32)

    # Define loss weights
    loss_weights = {
        'L_CE': 1.0, 
        'L_s': 1.0, 
        'L_e': 1.0, 
        'L_o': 0.5, 
        'L_seq': 0.5
    }

    # trained_model = train_model(model,
    #                             train_loader,
    #                             val_loader,
    #                             device,
    #                             num_epochs=30,
    #                             lr=5e-5,
    #                             loss_weights=loss_weights)

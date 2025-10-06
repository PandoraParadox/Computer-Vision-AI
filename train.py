import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from data.dataset import ClassificationDataset, improved_collate_fn
from models.backbone import ClassificationModel, ResNetClassifier, VGG11Classifier
from models.loss import ClassificationLoss
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'TRAIN_DIR': 'data/train',
    'VAL_SPLIT': 0.2,
    'EPOCHS': 100,
    'BATCH_SIZE': 4,  
    'LEARNING_RATE': 1e-5, 
    'IMG_SIZE': 128,  
    'NUM_CLASSES': 3,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SAVE_DIR': 'weights',
    'NUM_WORKERS': 2,  
    'DEBUG_MODE': False,
    'GRADIENT_CLIP': 5.0,  
    'WARMUP_EPOCHS': 5,
}


def get_learning_rate(epoch, base_lr, warmup_epochs):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        remaining_epochs = CONFIG['EPOCHS'] - warmup_epochs
        current_epoch = epoch - warmup_epochs
        return base_lr * 0.5 * (1 + np.cos(np.pi * current_epoch / remaining_epochs))

def train_one_epoch_stable(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    num_valid_batches = 0
    gradient_norms = []
    skipped_batches = 0
    
    current_lr = get_learning_rate(epoch, CONFIG['LEARNING_RATE'], CONFIG['WARMUP_EPOCHS'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    print(f"Learning rate: {current_lr:.8f}")
    
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        if len(imgs) == 0:
            print(f"Batch {batch_idx}: Empty batch, skipping")
            skipped_batches += 1
            continue
        
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=-1.0)
        imgs = torch.clamp(imgs, -3.0, 3.0)
        
        optimizer.zero_grad()
        
        try:
            outputs = model(imgs)
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Batch {batch_idx}: Invalid model outputs, skipping")
                skipped_batches += 1
                continue
            
            outputs = torch.clamp(outputs, -5.0, 5.0)
            
            loss, _, _, cls_loss = criterion(outputs, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Batch {batch_idx}: Invalid loss {loss.item()}, skipping")
                skipped_batches += 1
                continue
            
            loss.backward()
            
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)
            
            if total_norm > CONFIG['GRADIENT_CLIP']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRADIENT_CLIP'])
            
            optimizer.step()
            
            with torch.no_grad():
                accuracy = criterion.calculate_accuracy(outputs, targets)
                running_loss += loss.item()
                running_accuracy += accuracy
                num_valid_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, Acc={accuracy*100:.1f}%, "
                      f"GradNorm={total_norm:.3f}, LR={current_lr:.6f}")
        
        except RuntimeError as e:
            print(f"Batch {batch_idx}: Runtime error - {str(e)}, skipping")
            skipped_batches += 1
            if device == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    if num_valid_batches == 0:
        print("ERROR: No valid batches processed!")
        return 0.0, 0.0
    
    avg_loss = running_loss / num_valid_batches
    avg_accuracy = running_accuracy / num_valid_batches
    
    if gradient_norms:
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Valid batches: {num_valid_batches}/{len(train_loader)}")
        print(f"  Skipped batches: {skipped_batches}")
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy*100:.2f}%")
        print(f"  Gradient stats: mean={np.mean(gradient_norms):.3f}, max={np.max(gradient_norms):.3f}")
    
    return avg_loss, avg_accuracy

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            if len(imgs) == 0:
                continue
                
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            try:
                outputs = model(imgs)
                loss, _, _, _ = criterion(outputs, targets)
                
                accuracy = criterion.calculate_accuracy(outputs, targets)
                
                running_loss += loss.item()
                running_accuracy += accuracy
                num_batches += 1
                
            except Exception as e:
                print(f"Validation error: {str(e)}")
                continue
    
    if num_batches == 0:
        return 0.0, 0.0
    
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches
    
    return avg_loss, avg_accuracy

def main():
    os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
    
    full_dataset = ClassificationDataset(
        root_dir=CONFIG['TRAIN_DIR'],
        num_classes=CONFIG['NUM_CLASSES'],
        augment=True,
        img_size=CONFIG['IMG_SIZE']
    )

    from torch.utils.data import random_split
    val_size = int(len(full_dataset) * CONFIG['VAL_SPLIT'])
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_dataset = ClassificationDataset(
        root_dir=CONFIG['TRAIN_DIR'],
        num_classes=CONFIG['NUM_CLASSES'],
        augment=False,
        img_size=CONFIG['IMG_SIZE']
    )
    val_dataset.samples = [val_dataset.samples[i] for i in val_subset.indices]

    class_weights = full_dataset.get_class_weights()

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        collate_fn=improved_collate_fn,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True if CONFIG['DEVICE'] == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        collate_fn=improved_collate_fn,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True if CONFIG['DEVICE'] == 'cuda' else False
    )
    
    model = ClassificationModel(CONFIG['NUM_CLASSES'])
    model.to(CONFIG['DEVICE'])
    
    criterion = ClassificationLoss(
        num_classes=CONFIG['NUM_CLASSES'],
        label_smoothing=0.0,
        class_weights=class_weights.to(CONFIG['DEVICE'])
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['LEARNING_RATE'],
        weight_decay=1e-6,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    print(f"Bắt đầu huấn luyện với base LR: {CONFIG['LEARNING_RATE']}")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG['EPOCHS']):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{CONFIG['EPOCHS']}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_one_epoch_stable(
            model, train_loader, criterion, optimizer, epoch, CONFIG['DEVICE']
        )
        
        if train_loss == 0.0:
            print("Không có tiến độ huấn luyện, dừng lại...")
            torch.save(model.state_dict(), os.path.join(CONFIG['SAVE_DIR'], 'best_model.pth'))

            break
        
        if (epoch + 1) % 5 == 0 or epoch == CONFIG['EPOCHS'] - 1:
            val_loss, val_acc = validate_model(model, val_loader, criterion, CONFIG['DEVICE'])
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                best_model_path = os.path.join(CONFIG['SAVE_DIR'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'config': CONFIG,
                }, best_model_path)
                
                print(f"Mô hình tốt nhất được lưu! Val Acc: {val_acc*100:.2f}%")
            else:
                patience_counter += 1
        
        if patience_counter >= 10:
            print("Dừng sớm do không cải thiện")
            break
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CONFIG['SAVE_DIR'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'config': CONFIG,
            }, checkpoint_path)
    
    print(f"\nHuấn luyện hoàn tất! Độ chính xác validation tốt nhất: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nHuấn luyện bị gián đoạn bởi người dùng")
    except Exception as e:
        print(f"\nHuấn luyện thất bại: {e}")
        import traceback
        traceback.print_exc()
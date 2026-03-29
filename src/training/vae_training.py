import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def train_vae(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=100,
    device='cuda',
    lr=1e-3,
    patience=10,
    beta_start=0.001,  # Начальное значение beta для KL-дивергенции
    beta_end=0.01,     # Конечное значение beta
    warmup_epochs=10   # Количество эпох для постепенного увеличения beta
):
    """
    Улучшенная функция обучения VAE
    
    Основные улучшения:
    1. Постепенное увеличение веса KL-дивергенции (beta) в начале обучения
    2. Лучшая обработка размерностей
    3. Добавлена ранняя остановка
    4. Более информативный вывод
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon': [],
        'val_recon': [],
        'train_kl': [],
        'val_kl': [],
        'beta_values': []  # Для отслеживания изменения beta
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Рассчитываем текущее значение beta
        if epoch < warmup_epochs:
            # Линейное увеличение beta в течение первых warmup_epochs
            beta = beta_start + (beta_end - beta_start) * (epoch / warmup_epochs)
        else:
            beta = beta_end
        
        # ===== TRAINING =====
        model.train()
        train_losses = []
        train_recon_losses = []
        train_kl_losses = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            if isinstance(batch, (list, tuple)):
                x, _ = batch  # Извлекаем только данные, игнорируя метки
            else:
                x = batch
            
            x = x.to(device)
            
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x)
            
            # Проверяем размерности
            if recon_x.shape != x.shape:
                print(f"Размерности не совпадают: recon_x {recon_x.shape}, x {x.shape}")
                continue
            
            # Реконструкционная ошибка
            recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
            
            # KL-дивергенция
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            
            # Общий лосс
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            
            # Градиентный клиппинг для стабилизации обучения
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            train_recon_losses.append(recon_loss.item())
            train_kl_losses.append(kl_loss.item())
        
        if not train_losses:
            print("Нет данных для обучения, пропускаем эпоху")
            continue
            
        # Средние значения за эпоху
        avg_train_loss = np.mean(train_losses)
        avg_train_recon = np.mean(train_recon_losses)
        avg_train_kl = np.mean(train_kl_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['train_recon'].append(avg_train_recon)
        history['train_kl'].append(avg_train_kl)
        
        # ===== VALIDATION =====
        model.eval()
        val_losses = []
        val_recon_losses = []
        val_kl_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, _ = batch
                else:
                    x = batch
                
                x = x.to(device)
                
                recon_x, mu, logvar = model(x)
                
                # Проверяем размерности
                if recon_x.shape != x.shape:
                    continue
                
                recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                
                loss = recon_loss + beta * kl_loss
                
                val_losses.append(loss.item())
                val_recon_losses.append(recon_loss.item())
                val_kl_losses.append(kl_loss.item())
        
        if not val_losses:
            print("Нет данных для валидации, пропускаем эпоху")
            continue
            
        avg_val_loss = np.mean(val_losses)
        avg_val_recon = np.mean(val_recon_losses)
        avg_val_kl = np.mean(val_kl_losses)
        
        history['val_loss'].append(avg_val_loss)
        history['val_recon'].append(avg_val_recon)
        history['val_kl'].append(avg_val_kl)
        history['beta_values'].append(beta)
        
        # Обновляем learning rate на основе валидационной ошибки
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Beta: {beta:.4f}')
        print(f'  Train: Loss={avg_train_loss:.4f}, Recon={avg_train_recon:.4f}, KL={avg_train_kl:.4f}')
        print(f'  Val:   Loss={avg_val_loss:.4f}, Recon={avg_val_recon:.4f}, KL={avg_val_kl:.4f}')
        
        # Ранняя остановка
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save(model.state_dict(), 'best_vae_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Ранняя остановка на эпохе {epoch+1}")
                break
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load('best_vae_model.pth'))
    
    return history, model
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from test_models.diffusion import forward_add_noise
from test_models.I2IDifT import DifT_models
from time import time
from test_models.Improvedvae_tiny import ImprovedVAE
from ml_collections import config_dict

def train_engine(config):
    vae = ImprovedVAE()
    vae.load_state_dict(torch.load("./checkpoints/vae/VAE_tiny_f_psnr_35.52.pth"))
    vae.cuda()
    vae.eval()

    x_set = torch.load('./data/ixi/t1_pd_x_train.pt')
    y_set = torch.load('./data/ixi/t1_pd_y_train.pt')

    data_set = TensorDataset(x_set, y_set)
    print(f"Dataset contains {len(data_set):,} images")
    train_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)

    model = DifT_models['I2IDifT']()
    print(f"I2IDifT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.load_state_dict(torch.load(config.model_path))
    model.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    loss_fn = nn.L1Loss()

    model.train()
    train_steps = 0
    log_steps = 0
    running_loss = 0
    loss_weight = config.loss_scale
    epochs = config.epochs
    pre_loss = config.pre_loss
    start_time = time()

    for epoch in range(epochs):
        print(f"Beginning epoch {epoch}...")
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            t = torch.randint(0, 25, (x.shape[0],), device='cuda')

            with torch.no_grad():
                # Map input images to latent space
                x = vae.dift_encode(x, miu=True)
                y = vae.dift_encode(y, miu=True)

            if config.transpose:
                x = x.permute(1, 0, 2, 3)
                y = y.permute(1, 0, 2, 3)

            x, noise = forward_add_noise(x, t)
            pred_noise = model(x, t, y)
            loss = loss_weight * loss_fn(pred_noise, noise.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % config.log_every == 0:
                # Measure training speed:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all steps:
                avg_loss = torch.tensor(running_loss / log_steps, device='cuda') / config.loss_scale
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                if avg_loss < pre_loss:
                    pre_loss = avg_loss
                    path = './checkpoints/i2idift/dift_t1_pd_f_loss_{:.4f}.pth'.format(avg_loss)
                    torch.save(model.state_dict(), path)
                    print(f'model has saved, current loss is {avg_loss}')

        model.train()
    print("Done!")

def train_config():
    config = config_dict.ConfigDict()
    config.learning_rate = 1e-4
    config.log_every = 250
    config.loss_scale = 1
    config.epochs = 400
    config.transpose = False
    config.pre_loss = 0.035
    config.model_path = './checkpoints/i2idift/dift_init.pth'

    return config

if __name__ == "__main__":

    config = train_config()
    train_engine(config)
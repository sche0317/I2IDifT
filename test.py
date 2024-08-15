import torch
from torch.utils.data import TensorDataset, DataLoader
from ml_collections import config_dict
from test_models.I2IDifT import DifT_models
from test_models.Improvedvae_tiny import ImprovedVAE
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr

def backward_denoise(model, condition, config):
    T = 25
    betas = torch.linspace(0.0001, 0.02, T)  # (T,)
    alphas = 1 - betas  # (T,)
    alphas_cumprod = torch.cumprod(alphas, dim=-1)  # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
    alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]),
                                    dim=-1)  # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....]
    variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # denoise用的方差   (T,)

    x = torch.randn(size=(config.batch_size, 2, 28, 28)).cuda()
    alphas = alphas.cuda()
    alphas_cumprod = alphas_cumprod.cuda()
    variance = variance.cuda()
    y = condition.cuda()

    model.eval()
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            t = torch.full((x.size(0),), time).cuda()

            # 预测x_t时刻的噪音
            noise = model(x, t, y)

            # 生成t-1时刻的图像
            shape = (x.size(0), 1, 1, 1)
            mean = 1 / torch.sqrt(alphas[t].view(*shape)) * \
                   (
                           x -
                           (1 - alphas[t].view(*shape)) / torch.sqrt(1 - alphas_cumprod[t].view(*shape)) * noise
                   )
            if time != 0:
                x = mean + \
                    torch.randn_like(x) * \
                    torch.sqrt(variance[t].view(*shape))
            else:
                x = mean
            x = x.detach()
    return x

def compute_metrics(pred, real):
    pred = (pred+1) * 127.5
    pred = torch.clamp(pred, 0, 255)
    real = (real+1) * 127.5
    real = torch.clamp(real, 0, 255)
    real = real.cuda()
    average_psnr = psnr(pred, real)
    return average_psnr

def inference_config():
    config = config_dict.ConfigDict()
    config.batch_size = 1
    config.vae_path = './checkpoints/vae/VAE_tiny_f_psnr_35.52.pth'
    config.model_path = './checkpoints/i2idift/dift_t1_pd_f_loss_0.0301.pth'

    return config

def main(config):
    vae = ImprovedVAE()
    vae.load_state_dict(torch.load(config.vae_path))
    vae.cuda()
    vae.eval()

    model = DifT_models['I2IDifT']()
    print(f"I2IDifT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.load_state_dict(torch.load(config.model_path))
    model.cuda()
    model.eval()

    x_set = torch.load('./data/ixi/t1_pd_x_test.pt')
    y_set = torch.load('./data/ixi/t1_pd_y_test.pt')
    data_set = TensorDataset(x_set, y_set)
    print(f"Dataset contains {len(data_set):,} images")
    train_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)

    psnr_list = []
    for condition, target in train_loader:
        condition = condition.cuda()
        target = target.cuda()
        with torch.no_grad():
            condition = vae.dift_encode(condition, miu=True)
            latent_pred = backward_denoise(model, condition, config)
            pred = vae.decode(latent_pred)

        psnr_temp = compute_metrics(pred, target)
        psnr_list.append(psnr_temp)

    print(f'The mean value of psnr is {sum(psnr_list) / len(psnr_list):.2f}')


if __name__ == "__main__":

    config = inference_config()
    main(config)
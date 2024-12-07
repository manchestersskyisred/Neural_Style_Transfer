import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import cv2
import os


class StyleTransfer:
    def __init__(self, content_path, style_path, output_dir, image_size=(512, 512), device="cpu",
                 lam1=1e-3, lam2=1e7, lam3=5e-3, lr=0.01):
        self.device = torch.device(device)
        self.content_path = content_path
        self.style_path = style_path
        self.output_dir = output_dir
        self.image_size = image_size
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lr = lr

        # 预处理
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.unnormalize = lambda x: x * self.std + self.mu
        self.normalize = lambda x: (x - self.mu) / self.std

        self.transform = Compose([Resize(self.image_size), ToTensor()])

        # 加载 VGG 模型
        self.model = models.vgg19(pretrained=True).to(self.device)
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.eval()

        # 加载图像
        self.content_img = self._load_image(self.content_path)
        self.style_img = self._load_image(self.style_path)
        self.var_img = self.content_img.clone().requires_grad_(True)

        # 初始化特征提取器
        self.shunt_model = self._init_shunt_model()

    def _load_image(self, image_path):
        img = Image.open(image_path)
        img = self.normalize(self.transform(img).unsqueeze(0).to(self.device))
        return img

    def _save_image(self, tensor, filename):
        tensor = self.unnormalize(tensor.clone().detach()).squeeze(0)
        tensor = torch.clamp(tensor, 0, 1).permute(1, 2, 0).cpu().numpy() * 255
        tensor = tensor[..., ::-1].astype("uint8")  # 转为 BGR
        cv2.imwrite(os.path.join(self.output_dir, filename), tensor)

    def _init_shunt_model(self):
        class ShuntModel(nn.Module):
            def __init__(self, model, normalize, con_layers=None, sty_layers=None):
                super().__init__()
                self.module = model.features.eval()
                self.normalize = normalize  # 显式传递 normalize
                self.con_layers = con_layers if con_layers else [22]
                self.sty_layers = sty_layers if sty_layers else [1, 6, 11, 20, 29]
                for name, layer in self.module.named_children():
                    if isinstance(layer, nn.MaxPool2d):
                        self.module[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

            def forward(self, x):
                sty_feat_maps = []
                con_feat_maps = []
                x = self.normalize(x)  # 使用传递的 normalize
                for name, layer in self.module.named_children():
                    x = layer(x)
                    if int(name) in self.con_layers:
                        con_feat_maps.append(x)
                    if int(name) in self.sty_layers:
                        sty_feat_maps.append(x)
                return {"Con_features": con_feat_maps, "Sty_features": sty_feat_maps}

        return ShuntModel(self.model, self.normalize).to(self.device)

    def run(self, iterations=1000):
        optimizer = torch.optim.Adam([self.var_img], lr=self.lr)
        checkpoint_path = os.path.join(self.output_dir, "checkpoint.pth")

        # 计算风格目标的 Gram 矩阵
        sty_target = self.shunt_model(self.style_img)["Sty_features"]
        con_target = self.shunt_model(self.content_img)["Con_features"]
        gram_target = []
        for sty_feat in sty_target:
            b, c, h, w = sty_feat.size()
            gram_i = torch.mm(sty_feat.view(c, -1), sty_feat.view(c, -1).t()).div(c * h * w)
            gram_target.append(gram_i)

        # 开始训练
        for itera in range(iterations):
            optimizer.zero_grad()
            output = self.shunt_model(self.var_img)
            sty_output = output["Sty_features"]
            con_output = output["Con_features"]

            # 内容损失
            con_loss = sum(F.mse_loss(c, t) for c, t in zip(con_output, con_target))

            # 风格损失
            sty_loss = torch.tensor(0.0, device=self.device)
            for sty_feat, gram_t in zip(sty_output, gram_target):
                b, c, h, w = sty_feat.size()
                gram_o = torch.mm(sty_feat.view(c, -1), sty_feat.view(c, -1).t()).div(c * h * w)
                sty_loss += F.mse_loss(gram_o, gram_t)

            # 总变分损失
            TV_loss = (torch.sum(torch.abs(self.var_img[:, :, :, :-1] - self.var_img[:, :, :, 1:])) +
                       torch.sum(torch.abs(self.var_img[:, :, :-1, :] - self.var_img[:, :, 1:, :]))) / (b * c * h * w)

            # 总损失
            loss = self.lam1 * con_loss + self.lam2 * sty_loss + self.lam3 * TV_loss
            loss.backward()
            optimizer.step()

            # 日志输出
            if itera % 100 == 0:
                print(f"Iteration {itera}/{iterations}: Content Loss={con_loss.item():.4f}, "
                      f"Style Loss={sty_loss.item():.4f}, TV Loss={TV_loss.item():.4f}, Total Loss={loss.item():.4f}")
                self._save_image(self.var_img, f"transfer_{itera}.jpg")

            # 保存检查点
            if itera % 500 == 0:
                torch.save({
                    'iteration': itera,
                    'model_state_dict': self.var_img,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)

        # 保存最终结果
        final_filename = "transfer_final.jpg"
        self._save_image(self.var_img, final_filename)
        return os.path.join(self.output_dir, final_filename)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.var_img = checkpoint['model_state_dict'].requires_grad_(True)
        start_iteration = checkpoint['iteration']
        print(f"Resuming from iteration {start_iteration}")
        return start_iteration
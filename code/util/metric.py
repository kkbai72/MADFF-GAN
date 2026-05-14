import torch
import lpips
from pytorch_msssim import ssim, ms_ssim


class ImageMetrics:
    def __init__(self, device="cuda", lpips_net="alex"):
        """
        初始化指标计算类
        Args:
            device: 计算设备 ("cuda" 或 "cpu")
            lpips_net: LPIPS 网络类型 ("alex", "vgg", "squeeze")
        """
        self.device = torch.device(device)

        # 初始化 LPIPS 模型
        self.lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False).to(self.device)

    @staticmethod
    def _check_input(x: torch.Tensor, y: torch.Tensor):
        """检查输入是否合法"""
        assert x.shape == y.shape, "输入张量形状不一致"
        assert x.dim() == 4 and y.dim() == 4, "输入必须为 BCHW 格式"
        # assert (-1.0 <= x.min()) and (x.max() <= 1.0), "输入值范围需在 [-1, 1]"
        # assert (-1.0 <= y.min()) and (y.max() <= 1.0), "输入值范围需在 [-1, 1]"


    def _to_01_range(self, x: torch.Tensor) -> torch.Tensor:
        """将 [-1, 1] 范围转换为 [0, 1] 范围 (用于 PSNR/SSIM)"""
        return (x + 1.0) / 2.0

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        计算 PSNR、SSIM、LPIPS 指标
        Args:
            x: 预测图像张量，形状 [B,C,H,W]，范围 [-1, 1]
            y: 真实图像张量，形状 [B,C,H,W]，范围 [-1, 1]
        Returns:
            包含各指标均值的字典
        """
        self._check_input(x, y)
        # x=torch.clamp(x,-1,1)
        # y=torch.clamp(y,-1,1)
        x, y = x.to(self.device), y.to(self.device)

        # ---- 计算 PSNR ----
        # x_01 = self._to_01_range(x)  # 转换到 [0, 1]
        # y_01 = self._to_01_range(y)c
        x_01 = x  # 转换到 [0, 1]
        y_01 = y

        # y_01 = torch.clamp(y_01, 0, 1)

        # x_01=x_01*255
        # y_01=y_01*255

        mse = torch.mean((x_01 - y_01) ** 2, dim=(1, 2, 3))  # 按样本计算 MSE
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # PSNR = 20*log10(MAX/MSE), MAX=1

        # ---- 计算 SSIM ----
        ssim_val = ssim(
            x_01,
            y_01,
            data_range=1.0,  # 数据范围 [0, 1]
            size_average=False  # 返回每个样本的 SSIM
        )

        # ---- 计算 LPIPS ----
        # LPIPS 要求输入为 [-1, 1]，无需转换
        lpips_val = self.lpips_fn(x, y, normalize=False).squeeze()

        return psnr.mean().item(), ssim_val.mean().item(), lpips_val.mean().item()
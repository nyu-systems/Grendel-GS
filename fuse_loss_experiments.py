from utils.loss_utils import pixelwise_l1_with_mask, pixelwise_ssim_with_mask
import torch
import diff_gaussian_rasterization

def loss_torch(image, gt_image, mask, lambda_dssim=0.2):
    pixelwise_Ll1 = pixelwise_l1_with_mask(image, gt_image, mask)
    Ll1 = pixelwise_Ll1.sum()
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(image, gt_image, mask)
    ssim_loss = pixelwise_ssim_loss.sum()
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_loss)
    return loss

def kernel_fused_loss(image, gt_image, mask, lambda_dssim=0.2):
    # TODO: implement kernel fused loss. 
    # You need to implement a loss operator and its cuda kernel implementation including forward and backward, and call them here. 
    # You could implement in `diff_gaussian_rasterization/` repo and utilize its binding to call the cuda kernel, refer to `diff_gaussian_rasterization.GaussianRasterizer`. 

    pass

if __name__ == "__main__":
    # set random seed for reproducibility
    torch.manual_seed(0)

    image = torch.rand(3, 1000, 1000).cuda()
    gt_image = torch.rand(3, 1000, 1000).cuda()
    mask = torch.randint(0, 2, (1000, 1000), dtype=torch.bool).cuda()
    lambda_dssim = 0.2

    loss = loss_torch(image, gt_image, mask, lambda_dssim)
    print(loss.item())




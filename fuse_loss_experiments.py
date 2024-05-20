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
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(image, gt_image, mask)
    ssim_loss = pixelwise_ssim_loss.sum()
    Ll1 = diff_gaussian_rasterization.fused_loss(image, gt_image, mask, lambda_dssim)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_loss)
    return loss

def test_l1_naive(image, gt_image, mask, lambda_dssim=0.2):
    pixelwise_Ll1 = pixelwise_l1_with_mask(image, gt_image, mask)
    Ll1 = pixelwise_Ll1.sum()
    Ll1.backward()
    # print("naive l1 loss:", Ll1.item())
    return image.grad
    
def test_l1_fused(image, gt_image, mask, lambda_dssim=0.2):
    l1_fused = diff_gaussian_rasterization.fused_loss(image, gt_image, mask, lambda_dssim)
    l1_fused.backward()
    # print("fused loss:", l1_fused.item())
    return image.grad

def test_total_naive(image, gt_image, mask, lambda_dssim=0.2):
    pixelwise_Ll1 = pixelwise_l1_with_mask(image, gt_image, mask)
    Ll1 = pixelwise_Ll1.sum()
    pixelwise_Lssim = pixelwise_ssim_with_mask(image, gt_image, mask)
    Lssim = pixelwise_Lssim.sum()
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim)
    loss.backward()
    
    print("naive total loss:", loss.item())
    return image.grad

def test_total_fused(image, gt_image, mask, lambda_dssim=0.2):
    loss_fused = diff_gaussian_rasterization.fused_loss(image, gt_image, mask, lambda_dssim)
    loss_fused.backward()
    print("fused loss:", loss_fused.item())
    return image.grad

if __name__ == "__main__":
    # set random seed for reproducibility
    torch.manual_seed(1)

    image = torch.rand(3, 1000, 1000, requires_grad=True, device="cuda")
    image_fused = image.detach().clone().requires_grad_(True)
    gt_image = torch.rand(3, 1000, 1000).cuda()
    mask = torch.randint(0, 2, (1000, 1000), dtype=torch.bool).cuda()
    lambda_dssim = 0.2

    # loss = loss_torch(image, gt_image, mask, lambda_dssim)
    # loss_fused = kernel_fused_loss(image, gt_image, mask, lambda_dssim)
    # print(loss.item())
    # print(loss_fused.item())
    
    print("*************************")
    print("***** Naive Fused Loss *****")
    print("*************************")
    with torch.profiler.profile(
      activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
      ],
      schedule=torch.profiler.schedule(
        skip_first=5,
        wait=5,
        warmup=1,
        active=10)
    ) as p:
      for n in range(21):
        naive_grad = test_total_naive(image, gt_image, mask, lambda_dssim)
        p.step()
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    
    
    print("*************************")
    print("*** Fused Loss 1.0 ***")
    print("*************************")
    with torch.profiler.profile(
      activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
      ],
      schedule=torch.profiler.schedule(
        skip_first=5,
        wait=5,
        warmup=1,
        active=10)
    ) as p:
      for n in range(21):
        fused_grad = test_total_fused(image_fused, gt_image, mask, lambda_dssim)
        p.step()
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    
    naive_grad = test_total_naive(image, gt_image, mask, lambda_dssim)
    print("naive_grad: ", naive_grad)
    fused_grad = test_total_fused(image_fused, gt_image, mask, lambda_dssim)
    print("fused_grad: ", fused_grad)
    print("Same grad?", torch.allclose(naive_grad * (1.0 - lambda_dssim), fused_grad, rtol=0))

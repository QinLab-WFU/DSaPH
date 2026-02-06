import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models


class LabelNet(nn.Module):
    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of labels
        :param bit: number of the final binary code
        """
        super().__init__()
        self.cl_text = nn.Sequential(
            nn.Linear(y_dim, 4096), nn.ReLU(inplace=True), nn.BatchNorm1d(4096), nn.Linear(4096, bit), nn.Tanh()
        )

    def forward(self, x):
        y = self.cl_text(x)
        return y


class GradCAM2:
    """
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al.
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    """

    def __init__(self, model, target_layers):
        super().__init__()
        self.model = model
        self.target_layers = target_layers

        for layer in target_layers:
            layer.register_forward_hook(self.forward_hook)
            layer.register_full_backward_hook(self.backward_hook)
        # self.target_layers.register_forward_hook(self.forward_hook)
        # self.target_layers.register_full_backward_hook(self.backward_hook)

        self.activations = []
        self.grads = []

    def forward_hook(self, _, __, output):
        self.activations = []
        self.activations.append(output)

    def backward_hook(self, _, __, grad_output):
        self.grads = []
        self.grads.append(grad_output[0].detach())

    def calculate_cam(self, model_input):
        self.model.eval()
        # forward
        self.model.zero_grad()
        y_hat = self.model(model_input) # 对应分类头的输出 默认最后一个
        y_hat = y_hat[-1]

        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)
        # backward
        y_c = y_hat[:, max_class]
        y_c.backward(torch.ones_like(y_c))

        # get activations and gradients
        activations = self.activations[0]
        grads = self.grads[0]
        # calculate weights
        tmp = grads.reshape(grads.shape[0], grads.shape[1], -1)
        weights = torch.mean(tmp, dim=-1)
        weights = weights.reshape(grads.shape[0], grads.shape[1], 1, 1)

        cam = (weights * activations).sum(dim=1)
        cam[cam < 0] = 0
        cam = cam.view(cam.shape[0], 1, cam.shape[1], cam.shape[2])
        # cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=True)
        cam = torch.nn.functional.interpolate(cam, size=(224,224), mode="bilinear", align_corners=True)
        cam = cam / cam.max()

        return cam


def save_cam(image, cam, file_path):
    cam = cam.squeeze()
    # [H,W] -> [H,W,C]
    cam = np.expand_dims(cam, axis=2)
    heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    image = image / image.max()
    heatmap = heatmap / heatmap.max()

    result = 0.4 * heatmap + 0.6 * image
    result = result / result.max()
    final = (result * 255).astype(np.uint8)

    cv2.imwrite(file_path, final)


if __name__ == "__main__":
    from CF_VIT import cf_deit_small
    model = cf_deit_small(hash_bit=16, num_class=20)
    cam = GradCAM2(model,model.head)
    import  torch
    test = torch.randn(32,3,224,224).cuda()
    cam.calculate_cam(test)
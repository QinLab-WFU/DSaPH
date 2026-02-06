import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM  # pip install grad-cam
from pytorch_grad_cam.utils.image import preprocess_image
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

        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)

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
        y_hat = self.model(model_input)
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
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=True)
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
    cam_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).cuda()
    grad_cam = GradCAM(cam_model, [cam_model.layer4[-1]])
    grad_cam2 = GradCAM2(cam_model, cam_model.layer4[-1])

    img = cv2.imread("../_datasets/nuswide/images/0001_11496596.jpg")
    image = cv2.resize(img, (224, 224))
    input_tensor = preprocess_image(image).cuda()

    cam = grad_cam(input_tensor)
    cam2 = grad_cam2.calculate_cam(input_tensor).detach().cpu().numpy()
    print("cam:", cam.shape)
    print("cam2:", cam2.shape)

    save_cam(image, cam, "./CAM.jpg")
    save_cam(image, cam2, "./CAM2.jpg")

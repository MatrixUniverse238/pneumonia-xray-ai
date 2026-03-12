import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]


    def generate(self, input_tensor):

        output = self.model(input_tensor)

        class_idx = torch.argmax(output)

        self.model.zero_grad()

        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam / torch.max(cam)

        heatmap = cam.detach().numpy()

        return heatmap


# ---------------------------------------------------
# Image preprocessing for ResNet
# ---------------------------------------------------

def preprocess_image(image):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    image = Image.fromarray(image)

    tensor = transform(image).unsqueeze(0)

    return tensor


# ---------------------------------------------------
# Generate GradCAM
# ---------------------------------------------------

def generate_gradcam(model, image):

    model.eval()

    input_tensor = preprocess_image(image)

    # Last convolution layer of ResNet
    target_layer = model.layer4[-1]

    gradcam = GradCAM(model, target_layer)

    heatmap = gradcam.generate(input_tensor)

    return heatmap


# ---------------------------------------------------
# Overlay heatmap on original image
# ---------------------------------------------------

def overlay_heatmap(image, heatmap):

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    result = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return result
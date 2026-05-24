import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from src.utils.device import get_device
from src.models.colorization_network import ColorizationNetwork

from PIL import Image
from skimage import color


def colorize_image(
    image_path,
    model,
    kmeans,
    image_size=(128, 128),
    device="mps"
):


    rgb = np.array(
        Image.open(image_path)
        .convert("RGB")
        .resize(image_size)
    )



    lab = color.rgb2lab(rgb / 255.0)

    l_channel = lab[:, :, 0]


    x = torch.tensor(
        l_channel,
        dtype=torch.float32
    )

    x = x.unsqueeze(0).unsqueeze(0)
    x = x.to(device)



    model.eval()

    with torch.no_grad():

        logits = model(x)

        pred_classes = logits.argmax(dim=1)

    pred_classes = pred_classes.squeeze().cpu().numpy()



    ab = kmeans.cluster_centers_[pred_classes]



    reconstructed_lab = np.zeros(
        (
            l_channel.shape[0],
            l_channel.shape[1],
            3
        ),
        dtype=np.float32
    )

    reconstructed_lab[:, :, 0] = l_channel
    reconstructed_lab[:, :, 1:] = ab


    reconstructed_rgb = color.lab2rgb(
        reconstructed_lab
    )

    return (
        rgb,
        reconstructed_rgb,
        pred_classes
    )

device = get_device()

model = ColorizationNetwork(
    n_classes=32
)

checkpoint = torch.load(
    "notebooks/checkpoints/best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

kmeans = joblib.load(
    "test_kmeans.pkl"
)

original, colorized, classes = colorize_image(
    "data/test/Image_70.jpg",
    model,
    kmeans,
    device=device
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(
    color.rgb2gray(original),
    cmap="gray"
)
plt.title("Input L")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(colorized)
plt.title("Colorized")
plt.axis("off")

plt.show()


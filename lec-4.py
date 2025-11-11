import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms


# Load your model (replace with your actual model architecture)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # --- Convolutional Blocks ---
        # A conv block learns features from the image.
        self.conv_block1 = nn.Sequential(
            # Input: 3 channels (RGB), Output: 16 feature maps
            # Kernel size 3x3 is standard. Padding=1 keeps the image size the same.
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            ),  # output: 16x128x128
            nn.ReLU(),  # Activation function to introduce non-linearity
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # Downsamples the image by a factor of 2
        )

        self.conv_block2 = nn.Sequential(
            # Input: 16 channels from the previous block, Output: 32 feature maps
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Classifier Head ---
        # This part takes the learned features and makes a final prediction.
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flattens the 32x32x32 feature map into a single vector
            # The input features dimension calculation:
            # Our image starts at 128x128.
            # After first MaxPool: 128/2 = 64x64
            # After second MaxPool: 64/2 = 32x32
            # The number of channels from conv_block2 is 32.
            # So, the flattened size is 32 * 32 * 32.
            # we only need to do the math here for the first Linear layer.
            nn.Linear(in_features=32 * 32 * 32, out_features=128),
            nn.ReLU(),
            nn.Linear(
                in_features=128, out_features=num_classes
            ),  # Output layer has 2 neurons for our 2 classes
        )

    def forward(self, x):
        # This defines the forward pass: how data flows through the layers.
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


# Load model
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn_skin_lesion.pt", map_location="cpu"))
model.eval()

# Define image transforms
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# Prediction function
def predict(image):
    # Transform image
    img_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        # Use softmax for CrossEntropyLoss outputs
        probabilities = torch.softmax(output, dim=1)  # will be of shape [1, 2]
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item()

    # Map to class
    label = "Malignant" if pred_class == 1 else "Benign"

    return f"{label} (Confidence: {confidence:.2%})"


# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Skin Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Skin Lesion Classification",
    description="Upload a skin image to classify as Benign (0) or Malignant (1)",
)

if __name__ == "__main__":
    demo.launch(share=True)

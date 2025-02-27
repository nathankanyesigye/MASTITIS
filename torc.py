import torch
import torch.nn as nn

# Define the model architecture (same as during training)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = MyModel()

# Load the saved model weights
model.load_state_dict(torch.load('model_one.pth'))

# Set the model to evaluation mode
model.eval()

# Example input data (adjust as needed for your model)
input_data = torch.randn(1, 10)  # Example: 1 sample with 10 features

# Make predictions
with torch.no_grad():
    output = model(input_data)
    print("Model Prediction:", output)

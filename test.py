import torch
import numpy as np
import librosa

# Load the saved model or state_dict
model_path = "D://FCIH//GP//codes//_Detection//integrated//attack-agnostic-dataset-master//model2.pth"
loaded_object = torch.load(model_path)
loaded_object.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_object.to(device)

# Print the device of model parameters to ensure they are on the correct device
for param in loaded_object.parameters():
    print(param.device)

# Function to preprocess audio
def preprocess_audio(audio_path, sequence_length=17240):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Normalize audio
    audio /= np.max(np.abs(audio))

    # Convert audio to PyTorch tensor and add batch and channel dimensions
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    # Pad or truncate the sequence to the desired length
    if audio_tensor.shape[2] < sequence_length:
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, sequence_length - audio_tensor.shape[2]))
    else:
        audio_tensor = audio_tensor[:, :, :sequence_length]

    return audio_tensor.permute(0, 2, 1)  # Permute to [batch_size, sequence_length, 1]

# Path to the audio file you want to predict
audio_path = "D://FCIH//GP//Datasets//test//FreeVC.wav"

# Preprocess the audio
input_data = preprocess_audio(audio_path)

# Move input data to GPU if available
input_data = input_data.to(device)

# Ensure input data is on the correct device
print(input_data.device)

# Make prediction
with torch.no_grad():
    output = loaded_object(input_data)

# Convert the output to probabilities using sigmoid activation
probabilities = torch.sigmoid(output)

# Extract the predicted class (0 for fake, 1 for real)
predicted_class = (probabilities > 0.5).float()

# Print the prediction probabilities and class
print(f"Probabilities: {probabilities.item()}")
print(f"Predicted class: {'Real' if predicted_class.item() == 1 else 'Fake'}")














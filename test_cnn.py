import torch
from cnn_dataset import CommonVoiceSpectrogramDataset
from cnn_model import AgeAudioCNN

def test_pipeline():
    print("Testing data loader...")
    dataset = CommonVoiceSpectrogramDataset(
        metadata_path="cv_au_subset/metadata.csv",
        audio_dir="cv_au_subset/audio"
    )
    
    # Grab the very first sample
    mel_spec, label, client_id = dataset[0]
    print(f"Sample 0 - Mel-Spec Shape: {mel_spec.shape}, Label: {label}")
    
    # Needs a batch dimension (1, 1, Mels, Time)
    dummy_input = mel_spec.unsqueeze(0)
    print(f"CNN Input Shape: {dummy_input.shape}")
    
    model = AgeAudioCNN(num_classes=7)
    output = model(dummy_input)
    print(f"CNN Output Shape: {output.shape}")
    
if __name__ == "__main__":
    test_pipeline()

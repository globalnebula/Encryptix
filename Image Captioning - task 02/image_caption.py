import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from datasets import load_dataset
from collections import Counter
import nltk

nltk.download('punkt')

# Define the transformation to apply to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the Flickr8k dataset (example using Hugging Face datasets)
dataset = load_dataset('flickr8k', split='train')

# Custom Dataset class for Flickr8k dataset
class Flickr8kDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_path = sample['image']
        caption = sample['caption']

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, caption

# Create dataset and data loader
flickr8k_dataset = Flickr8kDataset(dataset, transform=transform)
data_loader = DataLoader(flickr8k_dataset, batch_size=32, shuffle=True, num_workers=4)

# Build vocabulary
def build_vocab(captions, threshold=5):
    counter = Counter()
    for caption in captions:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
    
    words = [word for word, count in counter.items() if count >= threshold]
    
    word_to_idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    idx_to_word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
    
    for i, word in enumerate(words, 4):
        word_to_idx[word] = i
        idx_to_word[i] = word
    
    return word_to_idx, idx_to_word

# Extract all captions from the dataset to build the vocabulary
all_captions = [sample['caption'] for sample in dataset]
word_to_idx, idx_to_word = build_vocab(all_captions)
vocab_size = len(word_to_idx)

# Define the Encoder CNN (using a pre-trained ResNet50 model)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# Define the Decoder RNN (using LSTM)
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = 20

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

# Hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 1
learning_rate = 0.001
num_epochs = 5

# Initialize the models
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(data_loader):
        # Convert captions to tensor indices
        caption_idxs = []
        lengths = []
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            caption_idx = [word_to_idx.get(word, word_to_idx['<unk>']) for word in tokens]
            caption_idx = [word_to_idx['<start>']] + caption_idx + [word_to_idx['<end>']]
            lengths.append(len(caption_idx))
            caption_idxs.append(torch.tensor(caption_idx))
        
        caption_idxs = nn.utils.rnn.pad_sequence(caption_idxs, batch_first=True, padding_value=word_to_idx['<pad>'])

        # Forward pass
        features = encoder(images)
        outputs = decoder(features, caption_idxs, lengths)
        loss = criterion(outputs, caption_idxs.reshape(-1))
        
        # Backward pass and optimization
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

# Save the model checkpoints
torch.save(encoder.state_dict(), 'encoder.ckpt')
torch.save(decoder.state_dict(), 'decoder.ckpt')

# Testing the model with sample generation
def generate_caption(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Generate a caption
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = encoder(image)
        sampled_ids = decoder.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Convert word IDs to words
        caption = []
        for word_id in sampled_ids:
            word = idx_to_word[word_id]
            caption.append(word)
            if word == '<end>':
                break
        caption = ' '.join(caption)
    
    return caption

image_path = 'test.jpg'
caption = generate_caption(image_path)
print(f'Generated Caption: {caption}')

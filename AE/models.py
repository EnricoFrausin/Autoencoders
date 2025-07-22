import torch
from torch import nn, optim

class AE_0(nn.Module):

    def __init__(self, input_dim, latent_dim, device, hidden_layers=1, decrease_rate=0.5,
                 activation_fn=nn.ReLU, output_activation_encoder=None, output_activation_decoder=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.number_of_hidden_layers = hidden_layers
        self.device = device
        self.activation_fn = activation_fn

        encoder_neurons_sizes = [input_dim ]

        for _ in range(self.number_of_hidden_layers):
            encoder_neurons_sizes.append(int(encoder_neurons_sizes[-1] * decrease_rate))

        encoder_neurons_sizes.append(latent_dim)

        encoder_layers = []
        for i in range(len(encoder_neurons_sizes) - 1):
            encoder_layers.append(nn.Linear(encoder_neurons_sizes[i], encoder_neurons_sizes[i + 1]))
            if i < len(encoder_neurons_sizes) - 2:
                encoder_layers.append(self.activation_fn())
            elif output_activation_encoder is not None:
                encoder_layers.append(output_activation_encoder())

        self.encoder = nn.Sequential(*encoder_layers).to(self.device)  #TOCHECK TO DEVICE

        decoder_neurons_sizes = list(reversed(encoder_neurons_sizes))
        decoder_layers = []
        for i in range(len(decoder_neurons_sizes) - 1):
            decoder_layers.append(nn.Linear(decoder_neurons_sizes[i], decoder_neurons_sizes[i + 1]))
            if i < len(decoder_neurons_sizes) - 2:
                decoder_layers.append(self.activation_fn())
            elif output_activation_decoder is not None:
                decoder_layers.append(output_activation_decoder())

        self.decoder = nn.Sequential(*decoder_layers).to(self.device)  #TOCHECK TO DEVICE

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, data):
        # Store original shape for reshaping output
        original_shape = data.shape
        
        if data.dim() > 2:
            data_flat = data.view(-1, self.input_dim)
        else:
            data_flat = data

        encoded = self.encode(data_flat)
        decoded = self.decode(encoded)

        # Reshape decoded output back to original input shape
        if original_shape != data_flat.shape:
            decoded = decoded.view(original_shape)

        return decoded

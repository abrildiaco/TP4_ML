import torch.nn as nn

# Autoencoder -------------------------------------------------------------------------------------------------------------
class Autoencoder(nn.Module):
    """
    Deterministic autoencoder implemented with PyTorch.
    The encoder maps the input into a latent representation, and the decoder reconstructs the input from that representation.

    Arguments:
        input_size (int): number of input features
        latent_dim (int): dimension of the latent representation
        hidden_nodes (list): number of neurons in each hidden encoder layer
        activation (str): activation function to use in hidden layers

    Returns:
        None
    """

    def __init__(self, input_size, latent_dim, hidden_nodes, activation = "relu"):
        super().__init__()

        self.activation_function = self._get_activation_function(activation)

        self.encoder = self._build_encoder(input_size, latent_dim, hidden_nodes)
        self.decoder = self._build_decoder(input_size, latent_dim, hidden_nodes)

    def _get_activation_function(self, activation):
        """
        Gets the PyTorch activation function selected by name.

        Arguments:
            activation (str): activation function name

        Returns:
            nn.Module: PyTorch activation function class
        """
        activation_functions = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
            "gelu": nn.GELU
        }

        if activation not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation}")

        return activation_functions[activation]

    def _build_encoder(self, input_size, latent_dim, hidden_nodes):
        """
        Builds the encoder network.
        The encoder maps the original input into the latent space.

        Arguments:
            input_size (int): number of input features
            latent_dim (int): dimension of the latent representation
            hidden_nodes (list): number of neurons in each hidden layer

        Returns:
            nn.Sequential: encoder network
        """
        layers = []
        previous_size = input_size

        for hidden_size in hidden_nodes:
            # Add a linear layer followed by the selected activation function
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(self.activation_function())
            previous_size = hidden_size

        # Final encoder layer maps the last hidden layer into the latent space
        layers.append(nn.Linear(previous_size, latent_dim))

        return nn.Sequential(*layers)

    def _build_decoder(self, input_size, latent_dim, hidden_nodes):
        """
        Builds the decoder network.
        The decoder maps the latent representation back to the original input space.

        Arguments:
            input_size (int): number of input features
            latent_dim (int): dimension of the latent representation
            hidden_nodes (list): number of neurons in each hidden layer

        Returns:
            nn.Sequential: decoder network
        """
        layers = []
        previous_size = latent_dim

        for hidden_size in reversed(hidden_nodes):
            # Mirror the encoder architecture to reconstruct the input
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(self.activation_function())
            previous_size = hidden_size

        # Final decoder layer returns to the original number of features
        layers.append(nn.Linear(previous_size, input_size))

        # Keep reconstructed pixel values in the range [0, 1]
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, X):
        """
        Reconstructs the input through the encoder and decoder.

        Arguments:
            X (torch.Tensor): input batch

        Returns:
            torch.Tensor: reconstructed batch
        """
        # Compress the input into the latent space
        z = self.encoder(X)

        # Reconstruct the input from the latent representation
        X_reconstructed = self.decoder(z)

        return X_reconstructed

    def encode(self, X):
        """
        Encodes input data into the latent space.

        Arguments:
            X (torch.Tensor): input batch

        Returns:
            torch.Tensor: latent representation
        """
        return self.encoder(X)

    def decode(self, z):
        """
        Decodes latent vectors into reconstructed inputs.

        Arguments:
            z (torch.Tensor): latent representation

        Returns:
            torch.Tensor: reconstructed batch
        """
        return self.decoder(z)
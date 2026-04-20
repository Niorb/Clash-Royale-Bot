import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerExtractor(BaseFeaturesExtractor):
    """
    A Transformer-based feature extractor that treats the board as a set of entities.
    It uses Multi-Head Self-Attention to evaluate relationships between units,
    buildings, and towers.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 65):
        # features_dim is 64 (transformer output) + 1 (elixir vector) = 65
        super().__init__(observation_space, features_dim)
        
        d_model = 64
        nhead = 4
        num_layers = 2
        
        # Entity embedding: Projects the 7 features into the transformer's latent space
        self.entity_embedding = nn.Linear(7, d_model)
        
        # Transformer Encoder: Processes the entities in parallel
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, observations):
        entities = observations["entities"] # Shape: (Batch, 40, 7)
        vector = observations["vector"]     # Shape: (Batch, 1)
        
        # 1. Generate padding mask (True for padding slots, False for real entities)
        # Feature at index 0 is "Is Active" (1.0 = Active, 0.0 = Padding)
        padding_mask = (entities[:, :, 0] == 0)
        
        # 2. Embed entities into higher-dimensional space
        x = self.entity_embedding(entities) # Shape: (Batch, 40, 64)
        
        # 3. Process through Transformer Encoder with attention masking
        # This allows entities to "attend" to each other based on proximity, type, and health.
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask) # Shape: (Batch, 40, 64)
        
        # 4. Global Average Pooling (ignoring padding tokens)
        # We multiply by the active mask (1.0/0.0) to zero out padding, then sum and divide.
        active_mask = entities[:, :, 0].unsqueeze(-1) # Shape: (Batch, 40, 1)
        active_count = active_mask.sum(dim=1).clamp(min=1) 
        
        pooled = (x * active_mask).sum(dim=1) / active_count # Shape: (Batch, 64)
        
        # 5. Concatenate pooled spatial features with the scalar elixir vector
        return th.cat([pooled, vector], dim=1) # Shape: (Batch, 65)

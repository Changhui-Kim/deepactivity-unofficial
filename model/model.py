import math
import torch
import torch.nn as nn

class MultiFeatureEmbedding(nn.Module):
    """
    Embeds multiple categorical features, concatenates them, and projects them
    to a final embedding dimension.
    """
    def __init__(self, num_features: int, vocab_sizes: list[int], embed_dim: int, output_dim: int):
        """
        Args:
            num_features (int): The number of categorical features to embed (e.g., 5 for activity chain).
            vocab_sizes (list[int]): A list of vocabulary sizes, one for each feature.
            embed_dim (int): The embedding dimension for each individual feature.
            output_dim (int): The final dimension after projecting the concatenated embeddings.
        """
        super().__init__()
        if not isinstance(vocab_sizes, list) or len(vocab_sizes) != num_features:
            raise ValueError(f"vocab_sizes must be a list of integers of length {num_features}.")
        self.embedders = nn.ModuleList([nn.Embedding(v_size, embed_dim, padding_idx=0) for v_size in vocab_sizes])
        self.linear_proj = nn.Linear(num_features * embed_dim, output_dim)
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A LongTensor of shape (seq_len, batch_size, num_features)
                              containing the integer indices for each feature.

        Returns:
            torch.Tensor: The projected tensor of shape (seq_len, batch_size, output_dim).
        """
        embeds = [self.embedders[i](x[..., i]) for i in range(self.num_features)]
        concatenated_embeds = torch.cat(embeds, dim=-1)
        projected_embeds = self.linear_proj(concatenated_embeds)
        return projected_embeds

class CombinedInputEmbedding(nn.Module):
    """
    Creates the combined embedded input for the Transformer model.
    """
    def __init__(self, h2: int, act_vocab_sizes: list[int], act_embed_dim: int,
                 person_vocab_sizes: list[int], person_embed_dim: int,
                 household_vocab_sizes: list[int], household_embed_dim: int, num_features_list: list[int]):
        """
        Args:
            h2 (int): The final embedding dimension for all parts (d_model).
            act_vocab_sizes (list[int]): Vocab sizes for the 5 activity chain features.
            act_embed_dim (int): Embedding dimension for each activity feature.
            person_vocab_sizes (list[int]): Vocab sizes for the 23 target person features.
            person_embed_dim (int): Embedding dimension for each person feature.
            household_vocab_sizes (list[int]): Vocab sizes for the 9 household member features.
            household_embed_dim (int): Embedding dimension for each household feature.
            num_features_list (list[int]): Number of features in activity, person, and household [5, 23, 9]
        """
        super().__init__()
        self.h2 = h2
        self.activity_embedder = MultiFeatureEmbedding(num_features_list[0], act_vocab_sizes, act_embed_dim, h2)
        self.person_embedder = MultiFeatureEmbedding(num_features_list[1], person_vocab_sizes, person_embed_dim, h2)
        self.household_embedder = MultiFeatureEmbedding(num_features_list[2], household_vocab_sizes, household_embed_dim, h2)
        self.sep_token = nn.Parameter(torch.randn(1, 1, h2))

    def forward(self, activity_chain: torch.Tensor, target_person: torch.Tensor, household_members: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the combined input embedding.
        Args:
            activity_chain (torch.Tensor): LongTensor of the previous activity chain. Shape: (t, n, 5).
            target_person (torch.Tensor): LongTensor of the target person's info. Shape: (1, n, 26).
            household_members (torch.Tensor): LongTensor of other household members' info. Shape: (h, n, 9).
        Returns:
            torch.Tensor: The combined and embedded input tensor. Shape: (t + 1 + h*2 + 1, n, h2).
        """
        n = activity_chain.size(1)
        activity_embed = self.activity_embedder(activity_chain)
        person_embed = self.person_embedder(target_person)
        household_embeds = self.household_embedder(household_members)
        sep = self.sep_token.expand(1, n, -1)
        h_members_list = torch.unbind(household_embeds, dim=0)
        tensors_to_cat = [person_embed]
        for member_embed in h_members_list:
            tensors_to_cat.append(sep)
            tensors_to_cat.append(member_embed.unsqueeze(0))
        tensors_to_cat.append(sep)
        tensors_to_cat.append(activity_embed)
        return torch.cat(tensors_to_cat, dim=0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ActivityTransformerEncoder(nn.Module):
    def __init__(self, h2: int, act_vocab_sizes: list[int], act_embed_dim: int,
                 person_vocab_sizes: list[int], person_embed_dim: int,
                 household_vocab_sizes: list[int], household_embed_dim: int,
                 nhead: int, nlayers: int, d_hid: int, dropout: float = 0.1):
        """
        Args:
            h2 (int): The final embedding dimension for all parts (d_model).
            act_vocab_sizes, act_embed_dim: Parameters for activity embedding.
            person_vocab_sizes, person_embed_dim: Parameters for person embedding.
            household_vocab_sizes, household_embed_dim: Parameters for household embedding.
            nhead (int): The number of heads in the multiheadattention models.
            nlayers (int): The number of sub-encoder-layers in the encoder.
            d_hid (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super().__init__()
        self.d_model = h2
        num_features_list = [len(act_vocab_sizes), len(person_vocab_sizes), len(household_vocab_sizes)]
        self.embedding_layer = CombinedInputEmbedding(h2, act_vocab_sizes, act_embed_dim, person_vocab_sizes, person_embed_dim, household_vocab_sizes, household_embed_dim, num_features_list)
        self.pos_encoder = PositionalEncoding(h2, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=h2, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.fixed_seq_len = 10         # 10 = P0,...,P4 + <S> * 5

    def _generate_causal_mask(self, sz: int, activity_start_idx: int) -> torch.Tensor:
        mask = torch.zeros((sz, sz), dtype=torch.bool)
        activity_len = sz - activity_start_idx
        if activity_len > 0:
            causal_part = torch.triu(torch.ones(activity_len, activity_len), diagonal=1).bool()
            mask[activity_start_idx:, activity_start_idx:] = causal_part
        return mask

    def forward(self, activity_chain: torch.Tensor, target_person: torch.Tensor, household_members: torch.Tensor, household_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the Transformer Encoder.
        Args:
            activity_chain (torch.Tensor): Shape (t, n, 5).
            target_person (torch.Tensor): Shape (1, n, 26).
            household_members (torch.Tensor): Shape (h, n, 9), where h is the max number of members.
            household_padding_mask (torch.Tensor, optional): Mask for padded household members. Shape (n, h).
        Returns:
            torch.Tensor: The encoded output (memory). Shape: (t + 1 + h*2 + 1, n, h2).
        """
        embedded_input = self.embedding_layer(activity_chain, target_person, household_members)
        seq_len, batch_size, _ = embedded_input.shape
        device = embedded_input.device
        src_mask = self._generate_causal_mask(seq_len, self.fixed_seq_len).to(device)
        src_key_padding_mask = None

        # activity_chain: [seq_len, batch_size, num_features]
        # create sequence padding mask
        pad_idx = 0
        # [seq_len, batch_size] -> [batch_size, seq_len]
        src_key_padding_mask = (activity_chain[:,:,0] == pad_idx).transpose(0, 1)

        if household_padding_mask is not None:
            person_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            h_mask = household_padding_mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, -1)
            activity_len = activity_chain.size(0)
            final_sep_and_activity_mask = torch.zeros((batch_size, 1 + activity_len), dtype=torch.bool, device=device)
            src_key_padding_mask = torch.cat([person_mask, h_mask, final_sep_and_activity_mask], dim=1)
        pos_encoded_input = self.pos_encoder(embedded_input * math.sqrt(self.d_model))
        return self.transformer_encoder(pos_encoded_input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

class ActivityTransformerDecoder(nn.Module):
    def __init__(self, h2: int, tgt_act_vocab_sizes: list[int], tgt_act_embed_dim: int,
                 nhead: int, nlayers: int, d_hid: int, dropout: float = 0.1):
        """
        Args:
            h2 (int): The final embedding dimension (d_model).
            tgt_act_vocab_sizes (list[int]): Vocab sizes for the 3 target activity features (Type, Start, End).
            tgt_act_embed_dim (int): Embedding dimension for each target activity feature.
            nhead (int): The number of heads in the multiheadattention models.
            nlayers (int): The number of sub-decoder-layers in the decoder.
            d_hid (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super().__init__()
        self.d_model = h2
        self.tgt_act_embedding = MultiFeatureEmbedding(num_features=3, vocab_sizes=tgt_act_vocab_sizes, embed_dim=tgt_act_embed_dim, output_dim=h2)
        self.pos_encoder = PositionalEncoding(h2, dropout)
        decoder_layers = nn.TransformerDecoderLayer(d_model=h2, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=nlayers)

        # Output heads
        self.type_predictor = nn.Linear(h2, tgt_act_vocab_sizes[0])
        self.start_time_predictor = nn.Linear(h2, tgt_act_vocab_sizes[1])
        self.end_time_predictor = nn.Linear(h2, tgt_act_vocab_sizes[2])

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, tgt_activity: torch.Tensor, memory: torch.Tensor,memory_key_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Transformer Decoder.
        Args:
            tgt_activity (torch.Tensor): The target activity sequence. Shape (tgt_len, n, 3).
            memory (torch.Tensor): The output from the encoder. Shape (src_len, n, h2).
            memory_key_padding_mask (torch.Tensor, optional): Padding mask from the encoder. Shape (n, src_len).
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Logits for Type, StartTime, and EndTime predictions.
                Each has shape (tgt_len, n, vocab_size_for_that_feature).
        """
        tgt_embedded = self.tgt_act_embedding(tgt_activity)
        tgt_len = tgt_embedded.size(0)
        device = tgt_embedded.device
        tgt_mask = self._generate_causal_mask(tgt_len, device)
        pos_encoded_tgt = self.pos_encoder(tgt_embedded * math.sqrt(self.d_model))
        # output = self.transformer_decoder(pos_encoded_tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        pad_idx = 0
        tgt_key_padding_mask = (tgt_activity[:,:,0] == pad_idx).transpose(0, 1)

        output = self.transformer_decoder(
            pos_encoded_tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        type_logits = self.type_predictor(output)
        start_time_logits = self.start_time_predictor(output)
        end_time_logits = self.end_time_predictor(output)
        return type_logits, start_time_logits, end_time_logits

class FullActivityTransformer(nn.Module):
    def __init__(self, h2: int, nhead: int, enc_layers: int, dec_layers: int, d_hid: int,
                 src_act_vocab: list[int], src_act_embed: int,
                 person_vocab: list[int], person_embed: int,
                 household_vocab: list[int], household_embed: int,
                 tgt_act_vocab: list[int], tgt_act_embed: int, dropout: float = 0.1):
        """
        Args:
            h2 (int): The main embedding dimension (d_model).
            nhead (int): Number of attention heads.
            enc_layers (int): Number of encoder layers.
            dec_layers (int): Number of decoder layers.
            d_hid (int): Dimension of the feedforward networks.
            src_act_vocab, src_act_embed: Vocab sizes and embedding dim for source activity.
            person_vocab, person_embed: Vocab sizes and embedding dim for person info.
            household_vocab, household_embed: Vocab sizes and embedding dim for household info.
            tgt_act_vocab, tgt_act_embed: Vocab sizes and embedding dim for target activity.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.encoder = ActivityTransformerEncoder(h2, src_act_vocab, src_act_embed, person_vocab, person_embed, household_vocab, household_embed, nhead, enc_layers, d_hid, dropout)
        self.decoder = ActivityTransformerDecoder(h2, tgt_act_vocab, tgt_act_embed, nhead, dec_layers, d_hid, dropout)

    def forward(self, src_activity: torch.Tensor, person_info: torch.Tensor, household_info: torch.Tensor, 
                tgt_activity: torch.Tensor, household_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass.
        Args:
            src_activity (torch.Tensor): Source activity chain. Shape (t, n, 5).
            person_info (torch.Tensor): Target person info. Shape (1, n, 26).
            household_info (torch.Tensor): Household members info. Shape (4, n, 9).
            tgt_activity (torch.Tensor): Target activity sequence for the decoder. Shape (tgt_len, n, 3).
            household_padding_mask (torch.Tensor, optional): Padding mask for household members. Shape (n, 4).
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Logits for Type, StartTime, and EndTime predictions.
        """
        memory = self.encoder(src_activity, person_info, household_info, household_padding_mask)
        # The same padding mask applies to the memory in the decoder
        # We need to construct the full padding mask for the decoder's memory_key_padding_mask
        batch_size = src_activity.size(1)
        device = src_activity.device
        full_padding_mask = None
        if household_padding_mask is not None:
            person_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            h_mask = household_padding_mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, -1)
            activity_len = src_activity.size(0)
            final_sep_and_activity_mask = torch.zeros((batch_size, 1 + activity_len), dtype=torch.bool, device=device)
            full_padding_mask = torch.cat([person_mask, h_mask, final_sep_and_activity_mask], dim=1).to(device)

        return self.decoder(tgt_activity, memory, memory_key_padding_mask=full_padding_mask)

import math
import torch
import torch.nn as nn

class MultiFeatureEmbedding(nn.Module):
    """
    Embeds multiple categorical features, concatenates them, and projects them
    to a final embedding dimension. Expects batch-first input [B, T, num_features]
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
            x (torch.Tensor): A LongTensor of shape [B, T, num_features]
                              containing the integer indices for each feature.

        Returns:
            torch.Tensor: The projected tensor of [B, T, output_dim]
        """
        embeds = [self.embedders[i](x[..., i]) for i in range(self.num_features)]   # each [B, T, E]
        concatenated_embeds = torch.cat(embeds, dim=-1)                             # [B, T, num_features*E]
        return self.linear_proj(concatenated_embeds)                                # [B, T, output_dim]

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
        self.activity_embedder  = MultiFeatureEmbedding(num_features_list[0], act_vocab_sizes, act_embed_dim, h2)
        self.person_embedder    = MultiFeatureEmbedding(num_features_list[1], person_vocab_sizes, person_embed_dim, h2)
        self.household_embedder = MultiFeatureEmbedding(num_features_list[2], household_vocab_sizes, household_embed_dim, h2)

        # learnable <SEP> token (broadcast to batch)
        self.sep_token = nn.Parameter(torch.randn(1, 1, h2))

    def forward(self, 
                activity_chain: torch.Tensor,   # [B, T_a, 5]
                target_person: torch.Tensor,    # [B, 1, Dp]
                household_members: torch.Tensor # [B, H, Dh]
                ) -> torch.Tensor:
        
        B = activity_chain.size(0)
        activity_embed      = self.activity_embedder(activity_chain)        # [B, T_a, h2]
        person_embed        = self.person_embedder(target_person)           # [B, 1, h2]
        household_embed     = self.household_embedder(household_members)    # [B, H, h2]
        sep = self.sep_token.expand(B, 1, -1)                               # [B, 1, h2]

        # interleave sep + each household member
        pieces = [person_embed]
        H = household_embed.size(1)
        for i in range(H):
            pieces.append(sep)
            pieces.append(household_embed[:, i:i+1, :])     # [B, 1, h2]
        pieces.append(sep)              # final sep before activities
        pieces.append(activity_embed)   # [B, T_a, h2]

        return torch.cat(pieces, dim=1) # [B, 1 + 2H + 1 + T_a, h2]
        # h_members_list = torch.unbind(household_embeds, dim=0)
        # tensors_to_cat = [person_embed]
        # for member_embed in h_members_list:
        #     tensors_to_cat.append(sep)
        #     tensors_to_cat.append(member_embed.unsqueeze(0))
        # tensors_to_cat.append(sep)
        # tensors_to_cat.append(activity_embed)
        # return torch.cat(tensors_to_cat, dim=0)

class PositionalEncoding(nn.Module):
    """ 
    Batch-first positional encoding. x: [B, T, d_model]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)                                       # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)                                               # [T, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)                                                      # [T, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)        # [1, T, d_model] broadcast over batch
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
        self.embedding_layer = CombinedInputEmbedding(
            h2, 
            act_vocab_sizes, act_embed_dim, 
            person_vocab_sizes, person_embed_dim, 
            household_vocab_sizes, household_embed_dim, 
            num_features_list)
        self.pos_encoder = PositionalEncoding(h2, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h2, nhead=nhead, dim_feedforward=d_hid, 
            dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        # Sequence Layout: Person(1) + [SEP, Member]*4 + Sep(1) + Activities(T)
        # For H=4 members, activities start at 1 + 2*4 + 1 = 10
        self.fixed_activity_start_idx = 10         # 10 = P0,...,P4 + <S> * 5

    def _generate_causal_mask(self, seq_len: int, activity_start_idx: int, device) -> torch.Tensor:
        # allow free attention in the prefix; causal only inside activity segment
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        act_len = seq_len - activity_start_idx
        if act_len > 0:
            causal = torch.triu(torch.ones((act_len, act_len), dtype=torch.bool, device=device), diagonal=1)
            mask[activity_start_idx:, activity_start_idx:] = causal
        return mask # bool

    def _build_src_key_padding_mask(self,
                                    batch_size: int,
                                    activity_len: int,
                                    household_padding_mask: torch.Tensor | None) -> torch.Tensor | None:
        # Build mask that matches the combined sequence length: 1 + 2H + 1 + T
        # Person: always valid (False). Household: duplicate per [SEP, member].
        # Final [SEP] and activities: valid (False). If you have padded activities, customize here.
        if household_padding_mask is None:
            return None
        device = household_padding_mask.device
        person_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)               # [B,1]
        # household_padding_mask: [B, H] -> expand to [B, 2H] for [SEP, member] pairs
        h_mask = household_padding_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(batch_size, -1)  # [B, 2H]
        final_sep_and_acts = torch.zeros((batch_size, 1 + activity_len), dtype=torch.bool, device=device)  # [B, 1+T]
        return torch.cat([person_mask, h_mask, final_sep_and_acts], dim=1)  # [B, 1+2H+1+T]

    def forward(self, 
                activity_chain: torch.Tensor,       # [B, T_a, 5]
                target_person: torch.Tensor,        # [B, 1, Dp]
                household_members: torch.Tensor,    # [B, H, Dh] (H=4 with padding)
                household_padding_mask: torch.Tensor | None = None # [B, H] bool
                ) -> torch.Tensor:
        
        x = self.embedding_layer(activity_chain, target_person, household_members)     # [B, S, h2]
        B, S, _ = x.shape
        device = x.device
        src_mask = self._generate_causal_mask(S, self.fixed_activity_start_idx, device=device)      # [S, S] bool
        src_kpm = self._build_src_key_padding_mask(B, activity_len=activity_chain.size(1),
                                                   household_padding_mask=household_padding_mask)   # [B, S] bool or None
        
        x = self.pos_encoder(x * math.sqrt(self.d_model))   # [B, S, h2]
        return self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_kpm) # [B, S, h2]

class ActivityTransformerDecoder(nn.Module):
    def __init__(self, h2: int, tgt_act_vocab_sizes: list[int], tgt_act_embed_dim: int,
                 nhead: int, nlayers: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = h2
        self.tgt_act_embedding = MultiFeatureEmbedding(num_features=3, vocab_sizes=tgt_act_vocab_sizes, embed_dim=tgt_act_embed_dim, output_dim=h2)
        self.pos_encoder = PositionalEncoding(h2, dropout)
        dec_layer = nn.TransformerDecoderLayer(d_model=h2, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=nlayers)

        # Output heads
        self.type_head  = nn.Linear(h2, tgt_act_vocab_sizes[0])
        self.start_head = nn.Linear(h2, tgt_act_vocab_sizes[1])
        self.end_head   = nn.Linear(h2, tgt_act_vocab_sizes[2])

    def _generate_causal_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, 
                tgt_activity: torch.Tensor,     # [B, T_tgt, 3]
                memory: torch.Tensor,           # [B, S_src, h2]
                memory_key_padding_mask: torch.Tensor | None = None # [B, S_src] bool
                ):
        B, T_tgt, _ = tgt_activity.shape
        device = tgt_activity.device
        t = self.tgt_embed(tgt_activity)                        # [B, T_tgt, h2]
        tgt_mask = self._generate_causal_mask(T_tgt, device)    # [T_tgt, T_tgt] bool

        # pad idx 0 in first feature marks padding time steps (teacher forcing)
        pad_idx = 0
        tgt_kpm = (tgt_activity[..., 0] == pad_idx)             # [B, T_tgt] bool

        t = self.pos_encoder(t * math.sqrt(self.d_model))
        out = self.transformer_decoder(
            t, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=memory_key_padding_mask
        )                                                       # [B, T_tgt, h2]

        return (
            self.type_head(out),        # [B, T_tgt, C_type]
            self.start_head(out),       # [B, T_tgt, C_time]
            self.end_head(out),         # [B, T_tgt, C_time]
        )

class FullActivityTransformer(nn.Module):
    def __init__(self, h2: int, nhead: int, enc_layers: int, dec_layers: int, d_hid: int,
                 src_act_vocab: list[int], src_act_embed: int,
                 person_vocab: list[int], person_embed: int,
                 household_vocab: list[int], household_embed: int,
                 tgt_act_vocab: list[int], tgt_act_embed: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = ActivityTransformerEncoder(h2, src_act_vocab, src_act_embed, person_vocab, person_embed, household_vocab, household_embed, nhead, enc_layers, d_hid, dropout)
        self.decoder = ActivityTransformerDecoder(h2, tgt_act_vocab, tgt_act_embed, nhead, dec_layers, d_hid, dropout)

    def forward(self, 
                src_activity: torch.Tensor,         # [B, T_src, 5]
                person_info: torch.Tensor,          # [B, 1, Dp]
                household_info: torch.Tensor,       # [B, H, Dh]
                tgt_activity: torch.Tensor,         # [B, T_tgt, 3] 
                household_padding_mask: torch.Tensor | None = None # [B, H] bool
                ):
        memory = self.encoder(src_activity, person_info, household_info, household_padding_mask)

        # Build memory padding mask to match combined sequence length
        B = src_activity.size(0)
        device = src_activity.device
        full_padding_mask = None
        if household_padding_mask is not None:
            person_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)                      # [B, 1]
            h_mask = household_padding_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(B, -1)          # [B, 2H]
            final_sep_and_activity = torch.zeros((B, 1 + src_activity.size(1)), dtype=torch.bool, device=device)    # [B, 1+T]
            full_padding_mask = torch.cat([person_mask, h_mask, final_sep_and_activity], dim=1)     # [B, S]

        return self.decoder(tgt_activity, memory, memory_key_padding_mask=full_padding_mask)

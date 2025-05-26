"""
Cross-Attention Module for CoWPro Model
Implements multi-head cross-attention for better support-query interaction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        """
        Multi-head cross-attention module
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [B, N_q, D] - Query features (from query image)
            key: [B, N_k, D] - Key features (from support image)
            value: [B, N_k, D] - Value features (from support image)
            mask: [B, N_q, N_k] - Attention mask (optional)
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model)
        
        output = self.w_o(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output, attn_weights
    
    def attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output, attn_weights


class SpatialCrossAttention(nn.Module):
    def __init__(self, in_channels=256, num_heads=8, dropout=0.1):
        """
        Spatial cross-attention for 2D feature maps
        Args:
            in_channels: Input channel dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SpatialCrossAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # Convolutional layers for Q, K, V projections
        self.conv_q = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_k = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)
        
        # Position encoding
        self.pos_encoding = PositionalEncoding2D(in_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_channels)
        
    def forward(self, query_feat, support_feat, support_mask=None):
        """
        Args:
            query_feat: [B, C, H, W] - Query feature map
            support_feat: [B, C, H, W] - Support feature map
            support_mask: [B, 1, H, W] - Support mask (optional)
        """
        B, C, H, W = query_feat.shape
        
        # Add positional encoding
        query_feat_pos = self.pos_encoding(query_feat)
        support_feat_pos = self.pos_encoding(support_feat)
        
        # Generate Q, K, V
        Q = self.conv_q(query_feat_pos)  # [B, C, H, W]
        K = self.conv_k(support_feat_pos)  # [B, C, H, W]
        V = self.conv_v(support_feat_pos)  # [B, C, H, W]
        
        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, num_heads, HW, head_dim]
        K = K.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, num_heads, HW, head_dim]
        V = V.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, num_heads, HW, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, HW, HW]
        
        # Apply support mask if provided
        if support_mask is not None:
            support_mask_flat = support_mask.view(B, 1, 1, H * W)  # [B, 1, 1, HW]
            support_mask_flat = support_mask_flat.repeat(1, self.num_heads, H * W, 1)  # [B, num_heads, HW, HW]
            scores = scores.masked_fill(support_mask_flat == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, HW, head_dim]
        
        # Reshape back to spatial format
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        
        # Final projection
        output = self.conv_out(attn_output)
        
        # Residual connection and layer norm
        output_flat = output.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        query_flat = query_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        output_norm = self.layer_norm(output_flat + query_flat)
        output = output_norm.permute(0, 2, 1).view(B, C, H, W)
        
        return output, attn_weights.mean(dim=1)  # Return averaged attention weights


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, temperature=10000):
        """
        2D positional encoding for spatial feature maps
        """
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        self.temperature = temperature
        
    def forward(self, tensor):
        """
        Args:
            tensor: [B, C, H, W]
        """
        B, C, H, W = tensor.shape
        
        # Create coordinate grids
        y_embed = torch.arange(H, dtype=torch.float32, device=tensor.device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=tensor.device).unsqueeze(0).repeat(H, 1)
        
        # Normalize coordinates
        y_embed = y_embed / (H - 1) * 2 - 1
        x_embed = x_embed / (W - 1) * 2 - 1
        
        # Generate positional encodings
        dim_t = torch.arange(self.channels // 4, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.channels)
        
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
        
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Pad if necessary
        if pos.shape[1] < self.channels:
            pos = F.pad(pos, (0, 0, 0, 0, 0, self.channels - pos.shape[1]))
        elif pos.shape[1] > self.channels:
            pos = pos[:, :self.channels]
        
        return tensor + pos


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels=256, num_heads=8, num_layers=2, dropout=0.1):
        """
        Multi-layer cross-attention fusion module
        Args:
            in_channels: Input channel dimension
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
        """
        super(CrossAttentionFusion, self).__init__()
        
        self.num_layers = num_layers
        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(in_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, query_feat, support_feat, support_mask=None):
        """
        Args:
            query_feat: [B, C, H, W] - Query feature map
            support_feat: [B, C, H, W] - Support feature map
            support_mask: [B, 1, H, W] - Support mask
        """
        enhanced_query = query_feat
        all_attn_weights = []
        
        # Apply multiple cross-attention layers
        for i, cross_attn in enumerate(self.cross_attn_layers):
            enhanced_query, attn_weights = cross_attn(enhanced_query, support_feat, support_mask)
            all_attn_weights.append(attn_weights)
        
        # Feature enhancement
        enhanced_query = self.feature_enhance(enhanced_query) + query_feat
        
        return enhanced_query, all_attn_weights

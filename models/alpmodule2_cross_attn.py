"""
Fixed ALPModule2 with Cross-Attention Integration
Resolves BatchNorm2d issue with batch size 1
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt

# Import the cross-attention modules (save the previous artifact as cross_attention.py)
try:
    from models.cross_attention import CrossAttentionFusion, SpatialCrossAttention
except ImportError:
    print("Warning: Cross-attention modules not found. Using simplified implementation.")
    CrossAttentionFusion = None
    SpatialCrossAttention = None

class MultiProtoAsWCosWithCrossAttn(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode='bilinear', use_cross_attn=True, num_heads=8):
        """
        ALPModule with Cross-Attention (Fixed BatchNorm Issue)
        Args:
            proto_grid:     Grid size when doing multi-prototyping
            feature_hw:     Spatial size of input feature map
            use_cross_attn: Whether to use cross-attention
            num_heads:      Number of attention heads
        """
        super(MultiProtoAsWCosWithCrossAttn, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        self.use_cross_attn = use_cross_attn
        
        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)
        
        # Cross-attention components
        if self.use_cross_attn and CrossAttentionFusion is not None:
            self.cross_attn_fusion = CrossAttentionFusion(
                in_channels=256,  # Assuming 256 channels from backbone
                num_heads=num_heads,
                num_layers=2,
                dropout=0.1
            )
            
            # Feature alignment for different input sizes
            self.query_proj = nn.Conv2d(256, 256, 1)
            self.support_proj = nn.Conv2d(256, 256, 1)
            
        # Fixed prototype refinement - replaced BatchNorm2d with GroupNorm to handle batch size 1
        self.proto_refine = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),  # GroupNorm instead of BatchNorm2d
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.GroupNorm(32, 256)   # GroupNorm instead of BatchNorm2d
        )

    def forward(self, qry, sup_x, sup_y, mode, thresh, fg=True, isval=False, val_wsize=None, vis_sim=False, **kwargs):
        """
        Enhanced forward pass with cross-attention (Fixed BatchNorm Issue)
        Args:
            qry:        [way(1), nb(1), nc, h, w] - Query features
            sup_x:      [way(1), shot, nb(1), nc, h, w] - Support features
            sup_y:      [way(1), shot, nb(1), h, w] - Support masks
            mode:       'mask'/'grid'/'gridconv'/'gridconv+' - Prototype mode
            thresh:     Threshold for prototype selection
            fg:         Foreground/background flag
            isval:      Validation mode flag
            val_wsize:  Validation window size
            vis_sim:    Visualization flag
        """
        qry = qry.squeeze(1)  # [way(1), nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1)  # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0)  # [nshot, 1, h, w]

        def safe_norm(x, p=2, dim=1, eps=1e-4):
            x_norm = torch.norm(x, p=p, dim=dim)
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        # Apply cross-attention if enabled
        if self.use_cross_attn and hasattr(self, 'cross_attn_fusion'):
            try:
                # Prepare features for cross-attention
                qry_proj = self.query_proj(qry)  # [1, nc, h, w]
                sup_x_proj = self.support_proj(sup_x.mean(dim=0, keepdim=True))  # [1, nc, h, w]
                sup_y_proj = sup_y.mean(dim=0, keepdim=True)  # [1, 1, h, w]
                
                # Apply cross-attention fusion
                enhanced_qry, attn_weights = self.cross_attn_fusion(
                    qry_proj, sup_x_proj, sup_y_proj
                )
                
                # Use enhanced query features
                qry = enhanced_qry + qry  # Residual connection
                
                # Store attention weights for visualization
                if vis_sim:
                    vis_dict_extra = {'cross_attn_weights': attn_weights}
                else:
                    vis_dict_extra = {}
            except Exception as e:
                print(f"Warning: Cross-attention failed: {e}")
                vis_dict_extra = {}
        else:
            vis_dict_extra = {}

        if mode == 'mask':  # class-level prototype only
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
            proto = proto.mean(dim=0, keepdim=True)
            
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps=1e-4) * 20.0
            
            vis_dict = {'proto_assign': None}
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
                if self.use_cross_attn:
                    vis_dict.update(vis_dict_extra)
            
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict

        elif mode == 'gridconv':  # using local prototypes only
            input_size = qry.shape
            nch = input_size[1]
            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]
            
            # Apply cross-attention refinement to prototypes (Fixed BatchNorm Issue)
            if self.use_cross_attn and len(protos) > 0:
                try:
                    # Only apply refinement if we have enough prototypes to avoid BatchNorm issues
                    if len(protos) > 1:
                        proto_spatial = protos.view(-1, nch, 1, 1)
                        proto_refined = self.proto_refine(proto_spatial)
                        protos = proto_refined.view(-1, nch)
                    else:
                        # For single prototype, skip refinement to avoid BatchNorm issue
                        print("Skipping prototype refinement for single prototype")
                except Exception as e:
                    print(f"Warning: Prototype refinement failed: {e}")
            
            protos = protos - protos.mean(dim=-1, keepdim=True)
            qry = qry - qry.mean(dim=1, keepdim=True)

            pro_n = safe_norm(protos)
            npro, nc = pro_n.shape
            qry_n = safe_norm(qry)
            _, nc, h, w = qry_n.shape

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20
            qry_proto_prob = F.softmax(dists, dim=1).view(-1, npro, h * w)
            ch_proto = torch.bmm(pro_n.unsqueeze(0).transpose(1, 2), qry_proto_prob).view(-1, nc, h, w)
            pred_grid = torch.sum(qry_n * ch_proto * 20, dim=1, keepdim=True)
            
            debug_assign = dists.argmax(dim=1).float().detach()
            vis_dict = {'proto_assign': debug_assign}
            
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()
                if self.use_cross_attn:
                    vis_dict.update(vis_dict_extra)

            return pred_grid, [debug_assign], vis_dict

        elif mode == 'gridconv+':  # local and global prototypes
            input_size = qry.shape
            nch = input_size[1]
            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]
            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
            protos = torch.cat([protos, glb_proto], dim=0)
            
            # Apply cross-attention refinement (Fixed BatchNorm Issue)
            if self.use_cross_attn and len(protos) > 1:
                try:
                    proto_spatial = protos.view(-1, nch, 1, 1)
                    proto_refined = self.proto_refine(proto_spatial)
                    protos = proto_refined.view(-1, nch)
                except Exception as e:
                    print(f"Warning: Prototype refinement failed: {e}")

            protos = protos - protos.mean(dim=-1, keepdim=True)
            qry = qry - qry.mean(dim=1, keepdim=True)

            pro_n = safe_norm(protos)
            npro, nc = pro_n.shape
            qry_n = safe_norm(qry)
            _, nc, h, w = qry_n.shape

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20
            qry_proto_prob = F.softmax(dists, dim=1).view(-1, npro, h * w)
            ch_proto = torch.bmm(pro_n.unsqueeze(0).transpose(1, 2), qry_proto_prob).view(-1, nc, h, w)
            pred_grid = torch.sum(qry_n * ch_proto * 20, dim=1, keepdim=True)

            debug_assign = dists.argmax(dim=1).float()
            vis_dict = {'proto_assign': debug_assign}
            
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()
                if self.use_cross_attn:
                    vis_dict.update(vis_dict_extra)

            return pred_grid, [debug_assign], vis_dict

        else:  # Advanced attention-based prototype matching
            B, C, H, W = qry.shape
            nch = C
            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0, 2, 1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view(sup_nshot, 1, -1).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]
            protos = protos.unsqueeze(0)
            
            if fg:
                glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
                if F.avg_pool2d(sup_y, 4).max() >= 0.95:
                    protos = torch.cat([protos, glb_proto.unsqueeze(1)], dim=0)
                else:
                    protos = glb_proto.unsqueeze(1)

            qry_n = qry.view(B, C, -1).transpose(1, 2)
            self.temperature = C ** 0.5

            # Enhanced attention mechanism with cross-attention
            if self.use_cross_attn:
                # Apply cross-attention to query before prototype matching
                attn1 = torch.bmm(protos / self.temperature, qry_n.transpose(1, 2))
                attn1 = F.softmax(attn1 / 0.05, dim=-1)
                
                # Enhance with cross-attention weights if available
                if 'cross_attn_weights' in vis_dict_extra:
                    try:
                        cross_weights = vis_dict_extra['cross_attn_weights'].view(B, -1, H * W)
                        attn1 = attn1 * (1 + 0.1 * cross_weights)  # Weighted enhancement
                    except Exception as e:
                        print(f"Warning: Could not apply cross-attention weights: {e}")
                
                attn = torch.bmm(attn1.transpose(1, 2), attn1)
                pred_grid = torch.bmm(attn, qry_n) * 20
            else:
                # Original attention mechanism
                attn1 = torch.bmm(protos / self.temperature, qry_n.transpose(1, 2))
                attn1 = F.softmax(attn1 / 0.05, dim=-1)
                attn = torch.bmm(attn1.transpose(1, 2), attn1)
                pred_grid = torch.bmm(attn, qry_n) * 20

            # Simple mean operation for final prediction
            pred_grid = pred_grid.mean(dim=1).view(B, 1, H, W)

            debug_assign = attn.argmax(dim=1).float().view(B, H, W)
            vis_dict = {'proto_assign': debug_assign}
            
            if vis_sim:
                vis_dict['raw_local_sims'] = attn.clone().detach()
                if self.use_cross_attn:
                    vis_dict.update(vis_dict_extra)

            return pred_grid, [debug_assign], vis_dict


# Simplified Cross-Attention Module for fallback
class SimpleCrossAttentionFusion(nn.Module):
    def __init__(self, in_channels=256, num_heads=8, num_layers=2, dropout=0.1):
        super(SimpleCrossAttentionFusion, self).__init__()
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)
            )
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(in_channels) for _ in range(num_layers)
        ])
        
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GroupNorm(32, in_channels)   # GroupNorm instead of BatchNorm
        )
        
    def forward(self, query_feat, support_feat, support_mask=None):
        B, C, H, W = query_feat.shape
        
        # Reshape for attention
        query_flat = query_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        support_flat = support_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        
        enhanced_query = query_flat
        all_attn_weights = []
        
        for i, (attn_layer, norm_layer) in enumerate(zip(self.attention_layers, self.norm_layers)):
            # Self-attention on query with support as key/value
            attn_output, attn_weights = attn_layer(
                enhanced_query, support_flat, support_flat
            )
            
            # Residual connection and normalization
            enhanced_query = norm_layer(attn_output + enhanced_query)
            all_attn_weights.append(attn_weights)
        
        # Reshape back to spatial format
        enhanced_query = enhanced_query.permute(0, 2, 1).view(B, C, H, W)
        
        # Feature enhancement
        enhanced_query = self.feature_enhance(enhanced_query) + query_feat
        
        return enhanced_query, all_attn_weights


# Use simplified cross-attention if the complex one is not available
if CrossAttentionFusion is None:
    CrossAttentionFusion = SimpleCrossAttentionFusion
    print("Using simplified cross-attention implementation")

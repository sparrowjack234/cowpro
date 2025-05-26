"""
Modified FewShotSeg Model with Cross-Attention Integration
Enhanced version of the main model with cross-attention mechanisms
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the modified ALP module (save previous artifacts appropriately)
from models.alpmodule2_cross_attn import MultiProtoAsWCosWithCrossAttn
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder, Encoder
from util.utils import get_tversky_loss
from pdb import set_trace
import pickle
import torchvision

# Prototype modes
FG_PROT_MODE = 'gridconv+'  # using both local and global prototype with cross-attention
BG_PROT_MODE = 'gridconv'   # using local prototype only

# Thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

class FewShotSegWithCrossAttn(nn.Module):
    """
    Enhanced ALPNet with Cross-Attention
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
        use_cross_attn:     Whether to use cross-attention (default: True)
        num_heads:          Number of attention heads (default: 8)
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, use_cross_attn=True, num_heads=8):
        super(FewShotSegWithCrossAttn, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.use_cross_attn = use_cross_attn
        self.num_heads = num_heads
        
        self.get_encoder(in_channels)
        self.get_cls()

    def get_encoder(self, in_channels):
        """Initialize the backbone encoder"""
        use_coco_init = self.config['use_coco_init']
        self.encoder = TVDeeplabRes101Encoder(use_coco_init)

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path)['model'], strict=False)
            print(f'###### Pre-trained model {self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier with cross-attention
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        
        assert self.config['cls_name'] == 'grid_proto'
        
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsWCosWithCrossAttn(
                proto_grid=[proto_hw, proto_hw], 
                feature_hw=self.config["feature_hw"],
                use_cross_attn=self.use_cross_attn,
                num_heads=self.num_heads
            )
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz=False):
        """
        Enhanced forward pass with cross-attention
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            isval: validation flag
            val_wsize: validation window size
            show_viz: return the visualization dictionary
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-way has not been implemented yet"
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        # Concatenate all images for batch processing
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        # Extract features using backbone
        img_fts = self.encoder(imgs_concat, low_level=False)
        fts_size = img_fts.shape[-2:]

        # Split features back into support and query
        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
            
        # Process masks
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad=True)
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        # Compute loss and outputs
        align_loss = 0
        outputs = []
        visualizes = []

        for epi in range(1):  # batch dimension, fixed to 1
            # Interpolate masks to feature size
            res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='bilinear') 
                                    for fore_mask_w in fore_mask], dim=0)  # [nway, ns, nb, nh', nw']
            res_bg_msk = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='bilinear') 
                                    for back_mask_w in back_mask], dim=0)  # [nway, ns, nb, nh', nw']

            scores = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []

            # Background prototype matching with cross-attention
            _raw_score, _, aux_attr = self.cls_unit(
                qry_fts, supp_fts, res_bg_msk, 
                mode=BG_PROT_MODE, 
                fg=False, thresh=BG_THRESH, 
                isval=isval, val_wsize=val_wsize, 
                vis_sim=show_viz
            )

            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            # Foreground prototype matching with cross-attention
            for way, _msk in enumerate(res_fg_msk):
                _raw_score, _, aux_attr = self.cls_unit(
                    qry_fts, supp_fts, _msk.unsqueeze(0), 
                    fg=True, 
                    mode=FG_PROT_MODE,
                    thresh=FG_THRESH, 
                    isval=isval, val_wsize=val_wsize, 
                    vis_sim=show_viz
                )

                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])
                    
                    # Store cross-attention visualizations if available
                    if self.use_cross_attn and 'cross_attn_weights' in aux_attr:
                        visualizes.append({
                            'cross_attn_weights': aux_attr['cross_attn_weights'],
                            'way': way,
                            'type': 'foreground'
                        })

            # Combine scores
            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            # Prototype alignment loss (if enabled)
            if self.config['align'] and self.training:
                try:
                    align_loss_epi = self.alignLoss(
                        qry_fts[:, epi], pred, supp_fts[:, :, epi],
                        fore_mask[:, :, epi], back_mask[:, :, epi]
                    )
                    align_loss += align_loss_epi
                except:
                    align_loss += 0

        # Format outputs
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim=1)
        bg_sim_maps = torch.stack(bg_sim_maps, dim=1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim=1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Enhanced prototype alignment loss with cross-attention
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  # 1 x N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        skip_ways = []
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2)  # added to nway(1) and nb(1)

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
                
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1]

                qry_pred_fg_msk = F.interpolate(
                    binary_masks[way + 1].float(), 
                    size=img_fts.shape[-2:], 
                    mode='bilinear'
                )

                qry_pred_bg_msk = F.interpolate(
                    binary_masks[0].float(), 
                    size=img_fts.shape[-2:], 
                    mode='bilinear'
                )

                scores = []

                # Background score with cross-attention
                _raw_score_bg, _, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts, 
                    sup_y=qry_pred_bg_msk.unsqueeze(-3), 
                    fg=False, mode=BG_PROT_MODE, thresh=BG_THRESH
                )
                scores.append(_raw_score_bg)

                # Foreground score with cross-attention
                _raw_score_fg, _, _ = self.cls_unit(
                    qry=img_fts, sup_x=qry_fts, 
                    sup_y=qry_pred_fg_msk.unsqueeze(-3), 
                    fg=True, mode=FG_PROT_MODE, thresh=FG_THRESH
                )
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                # Construct support ground truth
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0

                # Compute losses
                loss.append(F.cross_entropy(supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)
                loss.append(get_tversky_loss(supp_pred.argmax(dim=1, keepdim=True), supp_label[None, ...], 0.3, 0.7, 1.0) / n_shots / n_ways)

        return torch.sum(torch.stack(loss)) if loss else torch.tensor(0.0, device=qry_fts.device)

    def get_attention_maps(self, supp_imgs, fore_mask, back_mask, qry_imgs, val_wsize=2):
        """
        Extract attention maps for visualization and analysis
        """
        self.eval()
        with torch.no_grad():
            # Run forward pass with visualization enabled
            output, _, vis_data, assign_maps = self.forward(
                supp_imgs, fore_mask, back_mask, qry_imgs, 
                isval=True, val_wsize=val_wsize, show_viz=True
            )
            
            return {
                'predictions': output,
                'bg_sim_maps': vis_data[0],
                'fg_sim_maps': vis_data[1],
                'assign_maps': assign_maps
            }

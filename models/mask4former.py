import torch
import hydra
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from models.modules.helpers_3detr import GenericMLP
from torch.cuda.amp import autocast
from models.modules.attention import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from pytorch3d.ops import sample_farthest_points


class Mask4Former(nn.Module):
    def __init__(
        self,
        backbone,
        num_queries,
        num_heads,
        num_decoders,
        num_levels,
        sample_sizes,
        mask_dim,
        dim_feedforward,
        num_labels,
    ):
        super().__init__()
        self.backbone = hydra.utils.instantiate(backbone)
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.num_decoders = num_decoders
        self.num_levels = num_levels
        self.sample_sizes = sample_sizes
        sizes = self.backbone.PLANES[-5:]

        self.point_features_head = conv(
            self.backbone.PLANES[7], mask_dim, kernel_size=1, stride=1, bias=True, D=3
        )

        self.query_projection = GenericMLP(
            input_dim=mask_dim,
            hidden_dims=[mask_dim],
            output_dim=mask_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        self.mask_embed_head = nn.Sequential(
            nn.Linear(mask_dim, mask_dim), nn.ReLU(), nn.Linear(mask_dim, mask_dim)
        )

        self.bbox_embed_head = nn.Sequential(
            nn.Linear(mask_dim, mask_dim),
            nn.ReLU(),
            nn.Linear(mask_dim, mask_dim),
            nn.ReLU(),
            nn.Linear(mask_dim, 6),
            nn.Sigmoid(),
        )

        self.class_embed_head = nn.Linear(mask_dim, num_labels + 1)
        self.pos_enc = PositionEmbeddingCoordsSine(d_pos=mask_dim)
        self.temporal_pos_enc = PositionEmbeddingCoordsSine(d_in=1, d_pos=mask_dim)
        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        for hlevel in range(self.num_levels):
            self.cross_attention.append(
                CrossAttentionLayer(
                    d_model=mask_dim,
                    nhead=self.num_heads,
                )
            )
            self.lin_squeeze.append(nn.Linear(sizes[hlevel], mask_dim))
            self.self_attention.append(
                SelfAttentionLayer(
                    d_model=mask_dim,
                    nhead=self.num_heads,
                )
            )
            self.ffn_attention.append(
                FFNLayer(
                    d_model=mask_dim,
                    dim_feedforward=dim_feedforward,
                )
            )

        self.decoder_norm = nn.LayerNorm(mask_dim)

    def forward(self, x, raw_coordinates=None, is_eval=False):
        device = x.device
        all_features = self.backbone(x)
        point_features = self.point_features_head(all_features[-1])

        with torch.no_grad():
            coordinates = me.SparseTensor(features=raw_coordinates, coordinates=x.C, device=device)
        pos_encodings_pcd = self.get_pos_encs(coordinates)

        sampled_coords = []
        mins = []
        maxs = []
        for coords, feats in zip(x.decomposed_coordinates, coordinates.decomposed_features):
            _, fps_idx = sample_farthest_points(coords[None, ...].float(), K=self.num_queries)
            sampled_coords.append(feats[fps_idx.squeeze(0).long(), :3])
            mins.append(feats[:, :3].min(dim=0)[0])
            maxs.append(feats[:, :3].max(dim=0)[0])

        sampled_coords = torch.stack(sampled_coords)
        mins = torch.stack(mins)
        maxs = torch.stack(maxs)

        query_pos = self.pos_enc(sampled_coords.float(), input_range=[mins, maxs])
        query_pos = self.query_projection(query_pos)

        queries = torch.zeros_like(query_pos).permute((0, 2, 1))
        query_pos = query_pos.permute((2, 0, 1))

        predictions_class = []
        predictions_bbox = []
        predictions_mask = []

        for _ in range(self.num_decoders):
            for hlevel in range(self.num_levels):
                output_class, outputs_bbox, outputs_mask, attn_mask = self.mask_module(
                    queries, point_features, self.num_levels - hlevel
                )

                decomposed_feat = all_features[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                pcd_sizes = [pcd.shape[0] for pcd in decomposed_feat]
                curr_sample_size = max(pcd_sizes)

                if not is_eval:
                    curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])

                rand_idx, mask_idx = self.get_random_samples(pcd_sizes, curr_sample_size, device)

                batched_feat = torch.stack([feat[idx, :] for feat, idx in zip(decomposed_feat, rand_idx)])

                batched_attn = torch.stack([attn[idx, :] for attn, idx in zip(decomposed_attn, rand_idx)])

                batched_pos_enc = torch.stack(
                    [pos_enc[idx, :] for pos_enc, idx in zip(pos_encodings_pcd[hlevel], rand_idx)]
                )

                batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == curr_sample_size] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])

                src_pcd = self.lin_squeeze[hlevel](batched_feat.permute((1, 0, 2)))

                output = self.cross_attention[hlevel](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(self.num_heads, dim=0).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                output = self.self_attention[hlevel](
                    output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos
                )

                # FFN
                queries = self.ffn_attention[hlevel](output).permute((1, 0, 2))

                predictions_class.append(output_class)
                predictions_bbox.append(outputs_bbox)
                predictions_mask.append(outputs_mask)

        output_class, outputs_bbox, outputs_mask = self.mask_module(queries, point_features)
        predictions_class.append(output_class)
        predictions_bbox.append(outputs_bbox)
        predictions_mask.append(outputs_mask)

        return {
            "pred_logits": predictions_class[-1],
            "pred_bboxs": predictions_bbox[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_bbox, predictions_mask),
        }

    def mask_module(self, query_feat, point_features, num_pooling_steps=0):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
        outputs_bbox = self.bbox_embed_head(query_feat)

        output_masks = []

        for feat, embed in zip(point_features.decomposed_features, mask_embed):
            output_masks.append(feat @ embed.T)

        output_masks = torch.cat(output_masks)
        outputs_mask = me.SparseTensor(
            features=output_masks,
            coordinate_manager=point_features.coordinate_manager,
            coordinate_map_key=point_features.coordinate_map_key,
        )

        if num_pooling_steps != 0:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = me.SparseTensor(
                features=(attn_mask.F.detach().sigmoid() < 0.5),
                coordinate_manager=attn_mask.coordinate_manager,
                coordinate_map_key=attn_mask.coordinate_map_key,
            )

            return outputs_class, outputs_bbox, outputs_mask.decomposed_features, attn_mask

        return outputs_class, outputs_bbox, outputs_mask.decomposed_features

    def get_pos_encs(self, coordinates):
        pos_encodings_pcd = []

        for _ in range(self.num_levels + 1):
            pos_encodings_pcd.append([])

            for coords_batch in coordinates.decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        coords_batch[None, :, :3].float(), input_range=[scene_min[:, :3], scene_max[:, :3]]
                    )
                    tmp += self.temporal_pos_enc(
                        coords_batch[None, :, 3].float(), input_range=[scene_min[:, 3:4], scene_max[:, 3:4]]
                    )

                pos_encodings_pcd[-1].append(tmp.squeeze(0).permute((1, 0)))

            coordinates = self.pooling(coordinates)

        pos_encodings_pcd.reverse()

        return pos_encodings_pcd

    def get_random_samples(self, pcd_sizes, curr_sample_size, device):
        rand_idx = []
        mask_idx = []
        for pcd_size in pcd_sizes:
            if pcd_size <= curr_sample_size:
                # we do not need to sample
                # take all points and pad the rest with zeroes and mask it
                idx = torch.zeros(curr_sample_size, dtype=torch.long, device=device)
                midx = torch.ones(curr_sample_size, dtype=torch.bool, device=device)
                idx[:pcd_size] = torch.arange(pcd_size, device=device)
                midx[:pcd_size] = False  # attend to first points
            else:
                # we have more points in pcd as we like to sample
                # take a subset (no padding or masking needed)
                idx = torch.randperm(pcd_size, device=device)[:curr_sample_size]
                midx = torch.zeros(curr_sample_size, dtype=torch.bool, device=device)

            rand_idx.append(idx)
            mask_idx.append(midx)
        return rand_idx, mask_idx

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_bbox, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_bboxs": b, "pred_masks": c}
            for a, b, c in zip(outputs_class[:-1], outputs_bbox[:-1], outputs_seg_masks[:-1])
        ]

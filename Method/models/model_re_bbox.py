import torch
from models import XVLMBase, load_pretrained
import itertools

from models.llm_interface import SpatialLLMInterface
from utils.llm_utils import LLMCache

import torch.nn.functional as F

def compute_rela(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    len_x = x1 - x2
    len_y = y1 - y2

    # Horizontal relationship
    if abs(len_x) < 0.5 * w1:
        horizontal = 1  # 'Center'
    elif len_x > 0:
        horizontal = 0  # 'Left'
    else:
        horizontal = 2  # 'Right'

    # Vertical relationship
    if abs(len_y) < 0.5 * h1:
        vertical = 1  # 'middle'
    elif len_y > 0:
        vertical = 0  # 'upper'
    else:
        vertical = 2  # 'lower'

    return torch.tensor([horizontal, vertical])

def distillation_loss(student_logits, teacher_probs, temperature=4.0):
    teacher_probs_soft = teacher_probs ** (1.0 / temperature)
    teacher_probs_soft = teacher_probs_soft / teacher_probs_soft.sum(dim=-1, keepdim=True)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kd_loss = F.kl_div(student_log_probs, teacher_probs_soft, reduction='batchmean')
    return kd_loss * (temperature ** 2)

class Bench(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=True, use_spatial_loss=True)
        
        self.bbox_collector = self.BBoxCollector(self)   

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []
        
        if config.get('use_llm_for_spatial', False):
            self.llm_interface = SpatialLLMInterface(config['llm_config_path'])
            self.llm_cache = LLMCache()
            self.llm_loss_weight = config.get('llm_loss_weight', 0.01)
            self.distill_temperature = config.get('distill_temperature', 4.0)
            self.distill_weight = config.get('distill_weight', 0.5)
        else:
            self.llm_interface = None
            self.llm_cache = None
            self.llm_loss_weight = 0.0
            self.distill_temperature = 4.0
            self.distill_weight = 0.0

        
        

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        # msg = self.load_state_dict(state_dict, strict=False)

        model_dict = self.state_dict()
        matched_state_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                matched_state_dict[k] = v
            else:
                print(f"Skipping {k}: pretrained shape {v.shape} vs current {model_dict[k].shape if k in model_dict else 'missing'}")
        
        model_dict.update(matched_state_dict)
        msg = self.load_state_dict(model_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, idx=None, pair=None):
        device = image.device
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        # output_coord & target_bbox: 64, 4
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx)

        n = len(pair)

        if n == 0:

            return loss_itc, loss_itm
        else:
            loss_count = 0
            total_spatial_loss = 0
            loss_bb = 0
            size = 12 
            all_valid_bboxes = [] 

            for i in range(n):
                num = pair[i][0]


                new_image_embed = image_embeds[num].unsqueeze(0)

                new = image[num].unsqueeze(0)
                image_embeds_new, _ = self.get_vision_embeds(new)

                tt = image_embeds_new.clone()

                raw_image_embeds = tt[:, 1:, :]
                features_before_avgpool_reshaped = raw_image_embeds.reshape(1, size, size, 1024)

                feature_map = features_before_avgpool_reshaped.permute(0,3,1,2)
                
                sen_token = pair[i][1]
                sen_embeds = self.get_text_embeds(sen_token.input_ids, sen_token.attention_mask)

                output_coord = self.predict_bbox(new_image_embed, sen_embeds, sen_token.attention_mask)

                loss_bbox, loss_giou = self.get_bbox_loss(output_coord, pair[i][2].unsqueeze(0))
                loss_bb += (loss_bbox + loss_giou)
                # Update the BBoxCollector with the current bbox information from the pair


                bbox_info = {
                    'bbox': pair[i][2],  # bbox
                    'image_feature_map': feature_map,
                    'num': pair[i][0]  # num
                }
                spatial_result = self.bbox_collector.update_bbox(bbox_info)

                if spatial_result is not None: 
                    spatial_loss, spatial_logits = spatial_result
                    total_spatial_loss += spatial_loss
                    loss_count += 1
                    
                if self.llm_interface is not None:
                    bbox = pair[i][2]
                    
                    if bbox[0] > -99:
                        all_valid_bboxes.append({
                            'bbox': bbox.cpu().numpy(), 
                            'num': num,
                            'feature_map': feature_map
                        })

            loss_bb = 0.1 * loss_bb/n

            self.bbox_collector.collect_bbox = []
            self.bbox_collector.current_num = None
            
            distill_loss = 0.0
            if self.llm_interface is not None and len(all_valid_bboxes) >= 2:
                from collections import defaultdict
                groups = defaultdict(list)
                for info in all_valid_bboxes:
                    groups[info['num']].append(info)

                teacher_pairs = []   
                for num, items in groups.items():
                    if len(items) < 2:
                        continue
                    from itertools import combinations
                    for (info1, info2) in combinations(items, 2):
                        teacher_pairs.append((info1['bbox'], info2['bbox'], info1['feature_map'])) 

                if teacher_pairs:
                    teacher_probs_list = []
                    student_logits_list = []  
                    for (bbox1, bbox2, feat_map) in teacher_pairs:

                        bbox1_t = torch.tensor(bbox1, device=device)
                        bbox2_t = torch.tensor(bbox2, device=device)
                        student_logit = self.get_spatial_relation_logits(bbox1_t, bbox2_t, feat_map)   # [1,9]
                        student_logits_list.append(student_logit)

                        cache_key = (tuple(bbox1), tuple(bbox2))
                        cached = self.llm_cache.get(bbox1, bbox2) if self.llm_cache else None
                        if cached and 'probs' in cached:
                            teacher_probs = torch.tensor(cached['probs'], device=device)
                        else:
                            teacher_probs = self.llm_interface.predict_spatial_relation_logits(bbox1, bbox2)  # [9]
                            if self.llm_cache:
                                self.llm_cache.set(bbox1, bbox2, {
                                    'probs': teacher_probs.cpu().numpy().tolist(),
                                    'relation_class': torch.argmax(teacher_probs).item()
                                })
                        teacher_probs_list.append(teacher_probs)
                    
                    if teacher_probs_list and student_logits_list:
                        student_logits = torch.cat(student_logits_list, dim=0)       # [K, 9]
                        teacher_probs = torch.stack(teacher_probs_list, dim=0)       # [K, 9]

                        distill_loss = distillation_loss(student_logits, teacher_probs, temperature=self.distill_temperature)
                        distill_loss = distill_loss * self.distill_weight
            
            if loss_count > 0:
                loss_spatial = total_spatial_loss / loss_count
                total_loss_spatial = loss_spatial + distill_loss
                return loss_itc, loss_itm, loss_bb, total_loss_spatial
            else:
                return loss_itc, loss_itm, loss_bb
        
    class BBoxCollector:
        def __init__(self,parent):
            self.collect_bbox = []
            self.current_num = None
            self.parent = parent

        def update_bbox(self, bbox_info):
            new_num = bbox_info['num']


            if not self.collect_bbox:
                self.collect_bbox.append(bbox_info)
                self.current_num = new_num
                return


            if len(self.collect_bbox) == 1:
                if new_num == self.current_num:
                    self.collect_bbox.append(bbox_info)
                    return
                else:

                    self.collect_bbox = [bbox_info]
                    self.current_num = new_num
                    return

            if len(self.collect_bbox) == 2:
                if new_num == self.current_num:
                    self.collect_bbox.append(bbox_info)
                    loss, logits = self.calculate_loss(self.collect_bbox)
                    self.collect_bbox = []
                    return loss, logits  

                else:
                    loss, logits = self.calculate_loss(self.collect_bbox)
                    self.collect_bbox = []
                    self.collect_bbox.append(bbox_info)
                    self.current_num = new_num
                    return loss, logits

        def calculate_loss(self, bboxes):
            permutations = list(itertools.permutations(bboxes, 2))
            total_loss = 0.0
            logits_list = []
            for pair in permutations:
                target_bbox_A = pair[0]['bbox']
                target_bbox_B = pair[1]['bbox']
                feature_map = pair[0]['image_feature_map']

                hv = compute_rela(target_bbox_A, target_bbox_B)   # tensor [2]
                target_class = int(hv[1] * 3 + hv[0])            # 0-8

                logits = self.parent.get_spatial_relation_logits(target_bbox_A, target_bbox_B, feature_map)
                loss = F.cross_entropy(logits, torch.tensor([target_class], device=logits.device))
                total_loss = total_loss + loss
                logits_list.append(logits)
            avg_loss = total_loss / len(permutations)
            avg_logits = torch.stack(logits_list).mean(dim=0) if logits_list else None
            return avg_loss, avg_logits


        

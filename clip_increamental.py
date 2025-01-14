import copy
import random
import pdb


import clip
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from  .finetune import Finetune
from .ocm import kwargs

def shrink_cov(cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0], device=cov.device)
    alpha1 = 1
    alpha2 = 1
    cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
    return cov_

def sample(mean, cov, size, shrink=False):
    vec = torch.randn(size, mean.shape[-1], device=mean.device)
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = torch.linalg.cholesky(cov)
    vec = vec @ sqrt_cov.t()
    vec = vec + mean
    return vec

def get_class_ids_per_task(args):
    yield args.class_order[:args.initial_increment]
    for i in range(args.initial_increment, len(args.class_order), args.increment):
        yield args.class_order[i:i + args.increment]

def get_class_names(classes_names, class_ids_per_task):
    return [classes_names[class_id] for class_id in class_ids_per_task]

class ClassIncrementalCLIP(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(self, backbone, feat_dim, num_class, **kwargs)
        self.cfg = kwargs['cfg']
        self.prompt_template = kwargs['cfg'].prompt_template
        self.device = kwargs['device']
        self.classes_names = None
        model, self.transforms = clip.load(kwargs['cfg'].model_name, device=kwargs['device'], jit=kwargs['jit'])
        self.visual = model.visual
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        # pdb.set_trace()
        self.class_ids_per_task = list(get_class_ids_per_task(kwargs['cfg']))
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = torch.float16 if kwargs['cfg'].fp16 else torch.float32
        self.adapter = nn.Linear(512, 512, bias=False, device=kwargs['device'])
        self.clip_type = model.dtype

        # old adapter
        self.old_adapter = None
        self.old_edge_samples = []
        self.old_edge_samples_labels = []
        self.old_edge_samples_nearest_labels = []

        # class stat
        self.class_mean_list = []
        self.class_cov_list = []

        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = kwargs['cfg'].mix_bias

        #some trials
        self.batch_idx = 0

    def encode_text(self, text, prompt=False):
        x = self.token_embedding(text).type(self.clip_type)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.clip_type)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_image(self, image):
        # 确保输入数据类型与 self.visual 的权重类型一致
        image = image.to(self.clip_type)
        return self.visual(image)

    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.type(torch.float32)

    def forward(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        image = image.type(torch.float16)
        with torch.no_grad():
            text_features = self.encode_text(self.text_tokens)

        with torch.no_grad():
            image_features = self.encode_image(image)
            original_image_features = image_features.clone()
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            image_features = torch.cat([image_features, memory_data], dim=0)
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            image_features = torch.cat([image_features, edge_sample], dim=0)

        image_features = self.adapter(image_features.type(self.dtype).detach()).type(self.clip_type)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if edge_sample is not None:
            edge_sample_features = image_features[-edge_num:]
            image_features = image_features[:-edge_num]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t().type(image_features.dtype)

        probs = logits_per_image
        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.old_adapter(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, image_features, old_memory_feature, edge_sample_features
            return probs, image_features, old_memory_feature, text_features
        if ori_ima_f:
            if memory_data is not None:
                image_features = image_features[:-memory_data.shape[0]]
            return probs, original_image_features, image_features
        return probs, image_features, None, None

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_idx])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.queue_empty = True
        self.hard_pairs = None
        self.task_idx = task_idx
        if task_idx > 0:
            self.old_adapter = copy.deepcopy(self.adapter)
            dist_list = []
            for k, class_name_feature in enumerate(self.class_name_features[:-len(self.class_ids_per_task[task_idx])]):
                diff = torch.cdist(
                    self.class_name_features[-len(self.class_ids_per_task[task_idx]):].type(torch.float32),
                    class_name_feature.unsqueeze(0).type(torch.float32)).squeeze()
                dist_list.append(diff)
            dist_list = torch.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < self.cfg.threshold #is it right?
            indices = torch.nonzero(mask)
            self.hard_new_class = torch.unique(indices[:, 1]) + self.cfg.initial_increment + (
                        task_idx - 1) * self.cfg.increment
            num_hard_class = self.hard_new_class.shape[0]
            self.hard_pairs = indices
            self.hard_pairs[:, 1] = self.hard_pairs[:, 1] + self.cfg.initial_increment + (
                        task_idx - 1) * self.cfg.increment

        if task_idx > 0:
            random_class_order_list = list(range(self.cfg.initial_increment + (task_idx - 1) * self.cfg.increment))
            random.shuffle(random_class_order_list)
            self.random_class_order_list = random_class_order_list
        self.batch_idx = -1

    def get_old_edge_samples(self, batch_size):
        random_select = torch.randperm(self.old_edge_samples.shape[0])[:batch_size]
        return self.old_edge_samples[random_select], self.old_edge_samples_labels[random_select], \
        self.old_edge_samples_nearest_labels[random_select]

    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4 * torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            self.class_edge_distance.append((max_distance.mean() - max_distance.min(),
                                             max_distance.max() - max_distance.mean(), max_distance.mean()))
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def observe(self, data):
        self.batch_idx += 1
        inputs, targets = data['image'], data['label']

        if self.task_idx > 0:
            sg_inputs = []
            sg_targets = []
            # num of classes per batch. Ensure an epoch traverses all classes at least once.
            # For exemple, if there are 100 classes and 50 batches per epoch , there will be 2 classes per batch.
            if self.cfg.dataset == "cifar100" and self.cfg.increment == 5:
                list_for_one_batch = [self.random_class_order_list[self.batch_idx * 4 % len(self.random_class_order_list)],
                                      self.random_class_order_list[(self.batch_idx * 4 + 1) % len(self.random_class_order_list)],
                                      self.random_class_order_list[(self.batch_idx * 4 + 2) % len(self.random_class_order_list)],
                                      self.random_class_order_list[(self.batch_idx * 4 + 3) % len(self.random_class_order_list)]]
            elif self.cfg.dataset == "imagenet_R":
                list_for_one_batch = [self.random_class_order_list[self.batch_idx * 5 % len(self.random_class_order_list)],
                                      self.random_class_order_list[(self.batch_idx * 5 + 1) % len(self.random_class_order_list)],
                                      self.random_class_order_list[(self.batch_idx * 5 + 2) % len(self.random_class_order_list)],
                                      self.random_class_order_list[(self.batch_idx * 5 + 3) % len(self.random_class_order_list)],
                                      self.random_class_order_list[(self.batch_idx * 5 + 4) % len(self.random_class_order_list)]]
            else:
                list_for_one_batch = [self.random_class_order_list[self.batch_idx * 2 % len(self.random_class_order_list)],
                                      self.random_class_order_list[(self.batch_idx * 2 + 1) % len(self.random_class_order_list)]]
            for i in list_for_one_batch:
                sg_inputs.append(
                    sample(self.class_mean_list[i], self.class_cov_list[i], int(10 * self.cfg.beta), shrink=self.cfg.shrinkage))
                sg_targets.append(torch.ones(int(10 * self.cfg.beta), dtype=torch.long, device=self.device) * i)
            sg_inputs = torch.cat(sg_inputs, dim=0)
            sg_targets = torch.cat(sg_targets, dim=0)
            targets = torch.cat([targets, sg_targets], dim=0)

        if self.hard_pairs is not None and self.hard_pairs.shape[0] > 0:
            edge_sample = []
            edge_p_target = []
            edge_n_target = []
            for hard_pair in self.hard_pairs:
                edge_sample.append(
                    sample(self.class_mean_list[hard_pair[0]], self.class_cov_list[hard_pair[0]], int(20 * self.cfg.beta),
                           shrink=self.cfg.shrinkage))
                edge_p_target.append(torch.ones(int(20 * self.cfg.beta), dtype=torch.long, device=self.device) * hard_pair[0])
                edge_n_target.append(torch.ones(int(20 * self.cfg.beta), dtype=torch.long, device=self.device) * hard_pair[1])
            edge_sample = torch.cat(edge_sample, dim=0)
            edge_p_target = torch.cat(edge_p_target, dim=0)
            edge_n_target = torch.cat(edge_n_target, dim=0)
        if self.task_idx > 0:
            not_ini = True
        else:
            not_ini = False

        outputs, _, __, edge_sample_features = self.forward(inputs, memory_data=sg_inputs, not_ini=not_ini,
                                                     edge_sample=edge_sample, prompt=False)
        pred = torch.argmax(outputs, dim=1)
        acc = torch.sum(pred == targets).item()

        if self.task_idx > 0:
            if edge_sample is not None:
                edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                edge_target_features = self.class_name_features[edge_p_target].type(edge_sample_features.dtype)
                edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                edge_nearest_class_features = self.class_name_features[edge_n_target].type(edge_sample_features.dtype)
                edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(dim=-1,
                                                                                                             keepdim=True)
                loss_hinge = torch.relu(- (edge_sample_features * edge_target_features.clone().detach()).sum(-1) + (
                            edge_sample_features * edge_nearest_class_features.clone().detach()).sum(-1) + 0.1).mean()
        loss_c = torch.nn.functional.cross_entropy(outputs, targets.detach())
        if edge_sample is not None:
            return pred, acc, loss_c + loss_hinge
        else:
            return pred, acc, loss_c

    def mix_matrix(self):
        if self.old_adapter is not None:
            weight_new = self.adapter.weight.data
            weight_old = self.old_adapter.weight.data
            dist = (weight_new - weight_old).abs()
            U_old, S_old, V_old = torch.linalg.svd(weight_old)
            P_new = U_old.T @ weight_new
            dist = (P_new - torch.diag(S_old) @ V_old).abs()
            mask = dist / dist.max()
            mask += self.mix_b
            mask = torch.clamp(mask, max=1)
            right = P_new * mask + torch.diag(S_old) @ V_old * (1 - mask)
            weight = U_old @ right
            self.adapter.weight.data = weight
            return

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        sample_loader = DataLoader(train_loader, batch_size=128, shuffle=False, num_workers=self.cfg.num_workers)
        sample_data = []
        sample_target = []
        sample_after_adapt_feature = []
        print('analyze')
        for input, target, task_ids in tqdm(sample_loader):
            input, target = input.to(self.device), target.to(self.device)
            with torch.no_grad():
                _, ori_ima_feat, after_adapt_feature = self.forward(input, ori_ima_f=True)
            sample_data.append(ori_ima_feat)
            sample_target.append(target)
            sample_after_adapt_feature.append(after_adapt_feature)
        sample_target = torch.cat(sample_target, dim=0)
        sample_data = torch.cat(sample_data, dim=0)
        sample_after_adapt_feature = torch.cat(sample_after_adapt_feature, dim=0)
        self.analyze_mean_cov(sample_data, sample_target)
        self.mix_matrix()
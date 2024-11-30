import torch
import torch.nn.functional as F


def negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale):
    # Normalize embeddings
    batch_size = img_embs.shape[0]
    labels = torch.arange(batch_size, device=img_embs.device).long()

    img_text_similarity = logit_scale * img_embs @ text_embs.t()
    text_img_similarity = logit_scale * text_embs @ img_embs.t()
    img_negtext_similarity = logit_scale * img_embs @ neg_text_embs.t()

    preds_i2t = torch.cat((img_text_similarity, img_negtext_similarity), dim=-1).argmax(
        dim=-1
    )
    preds_t2i = img_text_similarity.t().argmax(dim=-1)
    acc_i2t = (preds_i2t == labels).float().mean().item()
    acc_t2i = (preds_t2i == labels).float().mean().item()
    accuracy = (acc_i2t + acc_t2i) / 2

    loss = (
        F.cross_entropy(
            torch.cat([img_text_similarity, img_negtext_similarity], dim=-1), labels
        )
        + F.cross_entropy(text_img_similarity, labels)
    ).div(2)
    return loss, accuracy


def tripletclip_loss(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale):
    loss_1, accuracy1 = negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale)
    loss_2, accuracy2 = negclip_loss(neg_img_embs, neg_text_embs, text_embs, logit_scale)

    loss = loss_1 + loss_2
    accuracy = (accuracy1 + accuracy2) / 2
    return loss, accuracy

def clip_loss(img_embs, text_embs, logit_scale):
    # Normalize embeddings
    batch_size = img_embs.shape[0]
    labels = torch.arange(batch_size, device=img_embs.device).long()

    img_text_similarity = logit_scale * img_embs @ text_embs.t()
    text_img_similarity = logit_scale * text_embs @ img_embs.t()

    preds_i2t = img_text_similarity.argmax(dim=-1)
    preds_t2i = text_img_similarity.argmax(dim=-1)
    acc_i2t = (preds_i2t == labels).float().mean().item()
    acc_t2i = (preds_t2i == labels).float().mean().item()
    accuracy = (acc_i2t + acc_t2i) / 2

    loss = (
        F.cross_entropy(img_text_similarity, labels)
        + F.cross_entropy(text_img_similarity, labels)
    ).div(2)
    return loss, accuracy

import webdataset as wds
import random
import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import clip

import io

def image_handler(data):
    return Image.open(io.BytesIO(data))

def text_handler(data):
    return data.decode('utf-8')

class TripletClipData(Dataset):
    def __init__(self, data_dir, transforms, tokenizer):
        dataset = (
            wds.WebDataset(
                os.path.join(data_dir, "shard-{0..750}.tar"),
                resampled=True,
                handler=wds.ignore_and_continue,
                nodesplitter=None,
            )
            .shuffle(1000)
            .decode(
                wds.handle_extension(".image", image_handler),
                wds.handle_extension(".neg_image", image_handler),
                wds.handle_extension(".caption", text_handler),
                wds.handle_extension(".neg_caption", text_handler),
                handler=wds.ignore_and_continue
            )
            .rename(image="image", neg_image="neg_image", caption="caption", neg_caption="neg_caption")
            .to_tuple("image", "caption", "neg_image", "neg_caption", "__key__")
            .with_length(1000000)
            .with_epoch(1000000)
        )

        self.dataset_length = len(dataset)
        self.dataset_iter = iter(dataset)

        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        image, caption, neg_image, neg_caption, key = next(self.dataset_iter)
        input_ids = self.tokenizer(
                caption
            )[0]
        negative_input_ids = self.tokenizer(
            neg_caption,
        )[0]
        try:
            image = self.transforms(
                    image.convert("RGB")
                )
            neg_image = self.transforms(
                    neg_image.convert("RGB")
                )
        except:
            return None
        return image, neg_image, input_ids, negative_input_ids

class MSCOCO(Dataset):
    def __init__(self, data_dir, transforms, tokenizer) -> None:
        super().__init__()
        
        self.transforms = transforms
        self.tokenizer = tokenizer

        self.data = []
        
        self.image_path = os.path.join(data_dir, "val2017")
        self.caption_path = os.path.join(
            data_dir, "annotations", "captions_val2017.json"
        )

        images = os.listdir(self.image_path)
        with open(self.caption_path, "r") as h:
            captions = json.load(h)["annotations"]

        for d in captions:
            self.data.append(
                (str(d["image_id"]).zfill(12) + ".jpg", d["caption"].strip())
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, caption = self.data[index]
        img_path = os.path.join(self.image_path, img_name)
        input_ids = self.tokenizer(
                caption
            )[0]
        try:
            image = self.transforms(Image.open(img_path).convert("RGB"))
        except:
            return None
        return image, input_ids

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataloader(data_dir, transform, tokenizer, negtype, train, batch_size=1024, val_data_dir=""):
    train_dataset = TripletClipData(data_dir, transform, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=12, collate_fn=collate_fn
    )

    val_dataloader = None
    if val_data_dir != "":
        val_dataset = MSCOCO(val_data_dir, transform, tokenizer)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=12, collate_fn=collate_fn
        )

    return train_dataloader, val_dataloader

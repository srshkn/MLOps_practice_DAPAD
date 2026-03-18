from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    DataLoaders,
    GrandparentSplitter,
    ImageBlock,
    Normalize,
    Resize,
    aug_transforms,
    get_image_files,
    imagenet_stats,
    parent_label,
)

# Конфигурация
BATCH_SIZE = 32


def get_dataloaders(path: str) -> DataLoaders:
    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name="train", valid_name="test"),
        get_y=parent_label,
        item_tfms=Resize(128),
        batch_tfms=aug_transforms(size=128) + [Normalize.from_stats(*imagenet_stats)],
    )

    return data_block.dataloaders(path, bs=BATCH_SIZE)

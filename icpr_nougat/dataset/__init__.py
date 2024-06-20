from dataset.donut_dataset import NougatDataset, NougatPadFixSizeCollectFn

def get_dataset(dataset_args):
    dataset_type = dataset_args.get("type")
    dataset = eval(dataset_type)(**dataset_args)
    return dataset
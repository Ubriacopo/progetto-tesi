from common.data.dataset import SpecMediaBasedDataset


class AMIGOSDataset(SpecMediaBasedDataset):
    pass


if __name__ == "__main__":
    dataset = AMIGOSDataset(dataset_spec_file="data/amigos.csv", )

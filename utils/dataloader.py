# use the data loader with the same length (使用固定顺序)
import torch


class BalancedDataLoaderIterator:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

        self.num_dataloaders = len(dataloaders)

        max_length = max(len(dataloader) for dataloader in dataloaders)

        length_list = [len(dataloader) for dataloader in dataloaders]
        print("data loader length:", length_list)
        print("max dataloader length:", max_length,
              "epoch iteration:", max_length * self.num_dataloaders)
        self.total_length = max_length * self.num_dataloaders
        self.current_iteration = 0
        self.probabilities = torch.ones(
            self.num_dataloaders, dtype=torch.float) / self.num_dataloaders
        # self.current_dataloader_index = 0  # 新增变量，用于跟踪当前使用的数据加载器索引

    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.current_iteration = 0
        # self.current_dataloader_index = 0  # 重置当前数据加载器索引
        return self

    def __next__(self):
        if self.current_iteration >= self.total_length:
            raise StopIteration

        chosen_index = torch.multinomial(self.probabilities, 1).item() #self.probabilities 初始化为均等概率，使用 torch.multinomial 方法从中随机选择一个数据加载器。这确保了每个数据加载器被选择的概率是均等的。
        # chosen_index = self.current_dataloader_index  # 按顺序选择数据加载器
        try:
            sample = next(self.iterators[chosen_index])
        except StopIteration: # 当某个数据加载器的数据用完时，迭代器会被重新初始化，确保在整个训练过程中每个数据加载器的数据都会被均匀使用。
            self.iterators[chosen_index] = iter(self.dataloaders[chosen_index])
            sample = next(self.iterators[chosen_index])

        self.current_iteration += 1
        # self.current_dataloader_index = (self.current_dataloader_index + 1) % self.num_dataloaders  # 更新当前数据加载器索引
        return sample, chosen_index

    def __len__(self):
        return self.total_length

    def generate_fake_samples_for_batch(self, dataloader_id, batch_size):
        if dataloader_id >= len(self.dataloaders) or dataloader_id < 0:
            raise ValueError("Invalid dataloader ID")

        dataloader = self.dataloaders[dataloader_id]
        iterator = iter(dataloader)

        try:
            sample_batch = next(iterator)
            fake_samples = []

            for sample in sample_batch:
                if isinstance(sample, torch.Tensor):
                    fake_sample = torch.zeros(
                        [batch_size] + list(sample.shape)[1:])
                    fake_samples.append(fake_sample)
                else:
                    pass

            return fake_samples, dataloader_id
        except StopIteration:
            return None

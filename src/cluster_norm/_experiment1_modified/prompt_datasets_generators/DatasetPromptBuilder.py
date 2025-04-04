from abc import ABC, abstractmethod

class DatasetPromptBuilder(ABC):
    def __init__(self, dataset_path: str, k: int):
        self.dataset_path = dataset_path
        self.k = k

    @abstractmethod
    def build(self):
        pass

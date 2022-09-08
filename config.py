

class Config:
    def __init__(self):
        self.data_dir = 'datasets'
        self.IMAGE_SIZE = 48
        self.BATCH_SIZE = 32
        self.isColor = True
        self.max_lr = 0.01
        self.min_lr = 0.001
        self.epochs = 100

        self.margin = 2.0

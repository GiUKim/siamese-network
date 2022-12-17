

class Config:
    def __init__(self):
        self.data_dir = '/home2/kgu/101_ObjectCategories_siam'
        self.predict_model_path = 'checkpoints/model_23_0.8261.pth' 
        self.predict_path = "/home2/200_NORMAL/CF/TRAIN_CF/BUS_TRUCK_DATA_ADD/motorbike"
        self.IMAGE_SIZE = 48
        self.BATCH_SIZE = 32
        self.isColor = True
        self.max_lr = 0.01
        self.min_lr = 0.001
        self.epochs = 100

        self.margin = 2.0

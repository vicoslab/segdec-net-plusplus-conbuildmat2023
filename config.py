class Config:
    GPU = None

    RUN_NAME = None

    DATASET = None  # KSDD, DAGM, STEEL, KSDD2, sccdnet
    DATASET_PATH = None

    EPOCHS = None

    LEARNING_RATE = None
    DELTA_CLS_LOSS = None

    BATCH_SIZE = None

    WEIGHTED_SEG_LOSS = None
    WEIGHTED_SEG_LOSS_P = None
    WEIGHTED_SEG_LOSS_MAX = None
    DYN_BALANCED_LOSS = None
    GRADIENT_ADJUSTMENT = None
    FREQUENCY_SAMPLING = True

    # Default values
    ARCHITECTURE = 'SegDecNet++'
    FOLD = None
    TRAIN_NUM = None
    NUM_SEGMENTED = None
    RESULTS_PATH = "./RESULTS"
    SPLITS_PATH = None

    VALIDATE = True
    VALIDATE_ON_TEST = True
    VALIDATION_N_EPOCHS = 5
    USE_BEST_MODEL = False

    ON_DEMAND_READ = False
    REPRODUCIBLE_RUN = None
    MEMORY_FIT = 1
    SAVE_IMAGES = True
    DILATE = 1

    # Auto filled
    INPUT_WIDTH = None
    INPUT_HEIGHT = None
    INPUT_CHANNELS = None

    BEST_MODEL_TYPE = "dec"
    AUGMENTATION = False
    USE_NEGATIVES = None
    VAL_NEG = None
    OPTIMIZER = "sgd"
    SCHEDULER = None
    HARD_NEG_MINING = None
    PXL_DISTANCE = 2
    THR_ADJUSTMENT = None
    SEG_BLACK = False
    BCE_LOSS_W = False
    TRAIN_SPLIT = None

    def init_extra(self):
        if self.WEIGHTED_SEG_LOSS and (self.WEIGHTED_SEG_LOSS_P is None or self.WEIGHTED_SEG_LOSS_MAX is None):
            raise Exception("You also need to specify p and scaling factor for weighted segmentation loss!")
        if self.NUM_SEGMENTED is None:
            raise Exception("Missing NUM_SEGMENTED!")
        if self.DATASET == 'KSDD':
            self.INPUT_WIDTH = 512
            self.INPUT_HEIGHT = 1408
            self.INPUT_CHANNELS = 1

            if self.TRAIN_NUM is None:
                raise Exception("Missing TRAIN_NUM for KSDD dataset!")
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for KSDD dataset!")
            if self.FOLD is None:
                raise Exception("Missing FOLD for KSDD dataset!")

        elif self.DATASET == 'DAGM':
            self.INPUT_WIDTH = 512
            self.INPUT_HEIGHT = 512
            self.INPUT_CHANNELS = 1
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for DAGM dataset!")
            if self.FOLD is None:
                raise Exception("Missing FOLD for DAGM dataset!")
        elif self.DATASET == 'STEEL':
            self.INPUT_WIDTH = 1600
            self.INPUT_HEIGHT = 256
            self.INPUT_CHANNELS = 1

            self.VALIDATE_ON_TEST = False
            self.USE_BEST_MODEL = True
            print("Will use best model according to validation loss, validation is not performed on test set!")
            if not self.ON_DEMAND_READ:
                print("Will use ON_DEMAND_READ even though it is set on False!")
                self.ON_DEMAND_READ = True
            if self.TRAIN_NUM is None:
                raise Exception("Missing TRAIN_NUM for STEEL dataset!")
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for STEEL dataset!")
        elif self.DATASET == 'KSDD2':
            self.INPUT_WIDTH = 448
            self.INPUT_HEIGHT = 448
            self.INPUT_CHANNELS = 3
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for KSDD2 dataset!")
        elif self.DATASET in ['crack_segmentation', 'sccdnet']:
            self.INPUT_WIDTH = 448
            self.INPUT_HEIGHT = 448
            self.INPUT_CHANNELS = 3
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for crack_segmentation dataset!")
        elif self.DATASET == 'CFD':
            self.INPUT_WIDTH = 480
            self.INPUT_HEIGHT = 320
            self.INPUT_CHANNELS = 3
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for CFD dataset!")
        elif self.DATASET == 'CRACK500':
            self.INPUT_WIDTH = 640
            self.INPUT_HEIGHT = 360
            self.INPUT_CHANNELS = 3
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for CRACK500 dataset!")
        elif self.DATASET == 'DeepCrack':
            self.INPUT_WIDTH = 544
            self.INPUT_HEIGHT = 384
            self.INPUT_CHANNELS = 3
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for DeepCrack dataset!")
        else:
            raise Exception('Unknown dataset {}'.format(self.DATASET))

    def merge_from_args(self, args):
        self.ARCHITECTURE = args.ARCHITECTURE
        self.GPU = args.GPU
        self.RUN_NAME = args.RUN_NAME
        self.DATASET = args.DATASET
        self.DATASET_PATH = args.DATASET_PATH
        self.EPOCHS = args.EPOCHS
        self.LEARNING_RATE = args.LEARNING_RATE
        self.DELTA_CLS_LOSS = args.DELTA_CLS_LOSS
        self.BATCH_SIZE = args.BATCH_SIZE
        self.WEIGHTED_SEG_LOSS = args.WEIGHTED_SEG_LOSS
        self.WEIGHTED_SEG_LOSS_P = args.WEIGHTED_SEG_LOSS_P
        self.WEIGHTED_SEG_LOSS_MAX = args.WEIGHTED_SEG_LOSS_MAX
        self.DYN_BALANCED_LOSS = args.DYN_BALANCED_LOSS
        self.GRADIENT_ADJUSTMENT = args.GRADIENT_ADJUSTMENT
        self.FREQUENCY_SAMPLING = args.FREQUENCY_SAMPLING
        self.NUM_SEGMENTED = args.NUM_SEGMENTED

        self.BEST_MODEL_TYPE = args.BEST_MODEL_TYPE
        self.AUGMENTATION = args.AUGMENTATION
        self.USE_NEGATIVES = args.USE_NEGATIVES
        self.VAL_NEG = args.VAL_NEG
        self.OPTIMIZER = args.OPTIMIZER
        self.SCHEDULER = args.SCHEDULER
        self.HARD_NEG_MINING = args.HARD_NEG_MINING
        self.PXL_DISTANCE = args.PXL_DISTANCE
        self.THR_ADJUSTMENT = args.THR_ADJUSTMENT
        self.SEG_BLACK = args.SEG_BLACK
        self.BCE_LOSS_W = args.BCE_LOSS_W
        self.TRAIN_SPLIT = args.TRAIN_SPLIT

        if args.FOLD is not None: self.FOLD = args.FOLD
        if args.TRAIN_NUM is not None: self.TRAIN_NUM = args.TRAIN_NUM
        if args.RESULTS_PATH is not None: self.RESULTS_PATH = args.RESULTS_PATH
        if args.VALIDATE is not None: self.VALIDATE = args.VALIDATE
        if args.VALIDATE_ON_TEST is not None: self.VALIDATE_ON_TEST = args.VALIDATE_ON_TEST
        if args.VALIDATION_N_EPOCHS is not None: self.VALIDATION_N_EPOCHS = args.VALIDATION_N_EPOCHS
        if args.USE_BEST_MODEL is not None: self.USE_BEST_MODEL = args.USE_BEST_MODEL
        if args.ON_DEMAND_READ is not None: self.ON_DEMAND_READ = args.ON_DEMAND_READ
        if args.REPRODUCIBLE_RUN is not None: self.REPRODUCIBLE_RUN = args.REPRODUCIBLE_RUN
        if args.MEMORY_FIT is not None: self.MEMORY_FIT = args.MEMORY_FIT
        if args.SAVE_IMAGES is not None: self.SAVE_IMAGES = args.SAVE_IMAGES
        if args.DILATE is not None: self.DILATE = args.DILATE

    def get_as_dict(self):
        params = {
            "ARCHITECTURE": self.ARCHITECTURE,
            "GPU": self.GPU,
            "DATASET": self.DATASET,
            "DATASET_PATH": self.DATASET_PATH,
            "EPOCHS": self.EPOCHS,
            "LEARNING_RATE": self.LEARNING_RATE,
            "DELTA_CLS_LOSS": self.DELTA_CLS_LOSS,
            "BATCH_SIZE": self.BATCH_SIZE,
            "WEIGHTED_SEG_LOSS": self.WEIGHTED_SEG_LOSS,
            "WEIGHTED_SEG_LOSS_P": self.WEIGHTED_SEG_LOSS_P,
            "WEIGHTED_SEG_LOSS_MAX": self.WEIGHTED_SEG_LOSS_MAX,
            "DYN_BALANCED_LOSS": self.DYN_BALANCED_LOSS,
            "GRADIENT_ADJUSTMENT": self.GRADIENT_ADJUSTMENT,
            "FREQUENCY_SAMPLING": self.FREQUENCY_SAMPLING,
            "FOLD": self.FOLD,
            "TRAIN_NUM": self.TRAIN_NUM,
            "NUM_SEGMENTED": self.NUM_SEGMENTED,
            "RESULTS_PATH": self.RESULTS_PATH,
            "VALIDATE": self.VALIDATE,
            "VALIDATE_ON_TEST": self.VALIDATE_ON_TEST,
            "VALIDATION_N_EPOCHS": self.VALIDATION_N_EPOCHS,
            "USE_BEST_MODEL": self.USE_BEST_MODEL,
            "ON_DEMAND_READ": self.ON_DEMAND_READ,
            "REPRODUCIBLE_RUN": self.REPRODUCIBLE_RUN,
            "MEMORY_FIT": self.MEMORY_FIT,
            "INPUT_WIDTH": self.INPUT_WIDTH,
            "INPUT_HEIGHT": self.INPUT_HEIGHT,
            "INPUT_CHANNELS": self.INPUT_CHANNELS,
            "SAVE_IMAGES": self.SAVE_IMAGES,
            "DILATE": self.DILATE,
            "BEST_MODEL_TYPE": self.BEST_MODEL_TYPE,
            "AUGMENTATION": self.AUGMENTATION,
            "USE_NEGATIVES": self.USE_NEGATIVES,
            "VAL_NEG": self.VAL_NEG,
            "OPTIMIZER": self.OPTIMIZER,
            "SCHEDULER": self.SCHEDULER,
            "HARD_NEG_MINING": self.HARD_NEG_MINING,
            "PXL_DISTANCE": self.PXL_DISTANCE,
            "THR_ADJUSTMENT": self.THR_ADJUSTMENT,
            "SEG_BLACK": self.SEG_BLACK,
            "BCE_LOSS_W": self.BCE_LOSS_W,
            "TRAIN_SPLIT": self.TRAIN_SPLIT
        }
        return params


def load_from_dict(dictionary):
    cfg = Config()

    cfg.ARCHITECTURE = dictionary.get("ARCHITECTURE", None)
    cfg.GPU = dictionary.get("GPU", None)
    cfg.DATASET = dictionary.get("DATASET", None)
    cfg.DATASET_PATH = dictionary.get("DATASET_PATH", None)
    cfg.EPOCHS = dictionary.get("EPOCHS", None)
    cfg.LEARNING_RATE = dictionary.get("LEARNING_RATE", None)
    cfg.DELTA_CLS_LOSS = dictionary.get("DELTA_CLS_LOSS", None)
    cfg.BATCH_SIZE = dictionary.get("BATCH_SIZE", None)
    cfg.WEIGHTED_SEG_LOSS = dictionary.get("WEIGHTED_SEG_LOSS", None)
    cfg.WEIGHTED_SEG_LOSS_P = dictionary.get("WEIGHTED_SEG_LOSS_P", None)
    cfg.WEIGHTED_SEG_LOSS_MAX = dictionary.get("WEIGHTED_SEG_LOSS_MAX", None)
    cfg.DYN_BALANCED_LOSS = dictionary.get("DYN_BALANCED_LOSS", None)
    cfg.GRADIENT_ADJUSTMENT = dictionary.get("GRADIENT_ADJUSTMENT", None)
    cfg.FREQUENCY_SAMPLING = dictionary.get("FREQUENCY_SAMPLING", None)
    cfg.FOLD = dictionary.get("FOLD", None)
    cfg.TRAIN_NUM = dictionary.get("TRAIN_NUM", None)
    cfg.NUM_SEGMENTED = dictionary.get("NUM_SEGMENTED", None)
    cfg.RESULTS_PATH = dictionary.get("RESULTS_PATH", None)
    cfg.VALIDATE = dictionary.get("VALIDATE", None)
    cfg.VALIDATE_ON_TEST = dictionary.get("VALIDATE_ON_TEST", None)
    cfg.VALIDATION_N_EPOCHS = dictionary.get("VALIDATION_N_EPOCHS", None)
    cfg.USE_BEST_MODEL = dictionary.get("USE_BEST_MODEL", None)
    cfg.ON_DEMAND_READ = dictionary.get("ON_DEMAND_READ", None)
    cfg.REPRODUCIBLE_RUN = dictionary.get("REPRODUCIBLE_RUN", None)
    cfg.MEMORY_FIT = dictionary.get("MEMORY_FIT", None)
    cfg.INPUT_WIDTH = dictionary.get("INPUT_WIDTH", None)
    cfg.INPUT_HEIGHT = dictionary.get("INPUT_HEIGHT", None)
    cfg.INPUT_CHANNELS = dictionary.get("INPUT_CHANNELS", None)
    cfg.SAVE_IMAGES = dictionary.get("SAVE_IMAGES", None)
    cfg.DILATE = dictionary.get("DILATE", None)
    cfg.BEST_MODEL_TYPE = dictionary.get("BEST_MODEL_TYPE", None)
    cfg.AUGMENTATION = dictionary.get("AUGMENTATION", None)
    cfg.USE_NEGATIVES = dictionary.get("USE_NEGATIVES", None)
    cfg.VAL_NEG = dictionary.get("VAL_NEG", None)
    cfg.OPTIMIZER = dictionary.get("OPTIMIZER", None)
    cfg.SCHEDULER = dictionary.get("SCHEDULER", None)
    cfg.HARD_NEG_MINING = dictionary.get("HARD_NEG_MINING", None)
    cfg.PXL_DISTANCE = dictionary.get("PXL_DISTANCE", None)
    cfg.THR_ADJUSTMENT = dictionary.get("THR_ADJUSTMENT", None)
    cfg.SEG_BLACK = dictionary.get("SEG_BLACK", None)
    cfg.BCE_LOSS_W = dictionary.get("BCE_LOSS_W", None)
    cfg.TRAIN_SPLIT = dictionary.get("TRAIN_SPLIT", None)

    return cfg

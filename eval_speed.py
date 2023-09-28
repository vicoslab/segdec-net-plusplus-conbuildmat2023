from config import Config
from end2end import End2End
from data.dataset_catalog import get_dataset
from utils import create_folder
import sys, os, torch

def get_config():
    gpu, run_name, dataset, dataset_path, eval_name = sys.argv[1:6]
    print(gpu, run_name, dataset, dataset_path, eval_name)
    # Konfiguracija
    configuration = Config()

    params = [i.replace('\n', '') for i in open(os.path.join('RESULTS', dataset, run_name, 'run_params.txt'), 'r')]

    for p in params:
        p, v = p.split(":")
        try:
            v = int(v)
        except:
            try:
                v = float(v)
            except:
                pass

        if v == 'True':
            v = True
        elif v == 'False':
            v = False
        elif v == "None":
            v = None

        setattr(configuration, p, v)

    configuration.RUN_NAME = run_name
    configuration.GPU = gpu
    configuration.DATASET_PATH = dataset_path
    configuration.SAVE_IMAGES = True
    configuration.ON_DEMAND_READ = True

    if 'ARCHITECTURE' in os.environ:
        configuration.ARCHITECTURE = os.environ['ARCHITECTURE']

    if len(sys.argv) >= 8:
        print(sys.argv)
        thr_adjust, seg_black = sys.argv[6:9]

        configuration.THR_ADJUSTMENT = float(thr_adjust) if thr_adjust != "None" else None
        configuration.SEG_BLACK = bool(seg_black)
    else:
        configuration.THR_ADJUSTMENT = None #0.9
        configuration.SEG_BLACK = False #True

    configuration.init_extra()

    return configuration, eval_name

if __name__ == "__main__":
    cfg, eval_name = get_config()

    # Model

    end2end = End2End(cfg=cfg)
    end2end._set_results_path()
    end2end.print_run_params()
    end2end.set_seed()
    device = end2end._get_device()
    model = end2end._get_model().to(device)
    end2end.set_dec_gradient_multiplier(model, 0.0)

    if len(sys.argv) > 8:
        path = os.path.join(end2end.model_path, str(sys.argv[6]))
        model.load_state_dict(torch.load(path, map_location=f"cuda:{cfg.GPU}"))
        end2end._log(f"Loading model state from {path}")
    else:
        end2end.reload_model(model=model, load_final=False)

    # Make new eval save folder

    end2end.run_path = os.path.join(end2end.cfg.RESULTS_PATH, cfg.DATASET, eval_name)
    end2end.outputs_path = os.path.join(end2end.run_path, "test_outputs")
    create_folder(end2end.run_path)
    create_folder(end2end.outputs_path)

    end2end._log(f"Dataset: {cfg.DATASET}, Path: {cfg.DATASET_PATH}")

    with torch.no_grad():
        validation_loader = get_dataset("VAL", end2end.cfg)
        end2end.eval_model_speed(device=device, model=model, eval_loader=validation_loader)

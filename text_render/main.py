import argparse
import multiprocessing as mp
import os
import time
from multiprocessing.context import Process
from pathlib import Path
import cv2
import numpy as np

from loguru import logger
from tqdm import tqdm

from text_renderer.config import import_module_from_file, GenerateCfgV2
from text_renderer.render import Render

cv2.setNumThreads(1)

STOP_TOKEN = "kill"

# each child process will initialize Render in process_setup
render: Render


class DBWriterProcess(Process):
    def __init__(
        self,
        data_queue,
        save_dir: Path,
        generator_cfg: GenerateCfgV2,
        jpg_quality: int = 95
    ):
        super().__init__()
        self.data_queue = data_queue
        self.generator_cfg = generator_cfg
        self.save_dir = save_dir
        self.jpg_quality = jpg_quality

    def img_write(self, name: str, image: np.ndarray):
        img_path = os.path.join(self.save_dir, name + ".jpg")
        cv2.imwrite(img_path, image, self.encode_param())
    
    def encode_param(self):
        return [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
    
    def run(self):
        save_name = self.generator_cfg.save_image_name
                
        try:
            with open(self.save_dir.joinpath(save_name).with_suffix('.txt'), 'w', encoding='utf-8') as f:                
                start = time.time()
                
                while True:
                    m = self.data_queue.get()
                    if m == STOP_TOKEN:
                        logger.info("DBWriterProcess receive stop token")
                        break
                    
                    # Set name for the generated image
                    self.img_write(save_name, m["image"])                    
                    f.write(m["label"])
                    
                    logger.info(
                        f"sample {save_name} complete after {time.time() - start:.1f} s"
                    )
                    start = time.time()
                                           
        except Exception as e:
            logger.exception("DBWriterProcess error")
            raise e


def generate_img(data_queue):
    data = render()
    if data is not None:
        data_queue.put({"image": data[0], "label": data[1]})


def process_setup(*args):
    global render
    import numpy as np

    # Make sure different process has different random seed
    np.random.seed()

    render = Render(args[0])
    logger.info(f"Finish setup image generate process: {os.getpid()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='/media/duyla4/DATA/dataset/PROJECT_DATASET/ocr_tool/text_renderer/config/mozambique_synthesize/config.py', help="python file path")
    parser.add_argument("--dataset_path", type=str, default='/media/duyla4/DATA/dataset/PROJECT_DATASET/EKYC/MOVITEL/CARD/final_data/sync_corpus.txt')
    parser.add_argument("--root_dir", type=str, default='/media/duyla4/DATA/dataset/PROJECT_DATASET/EKYC/MOVITEL/CARD/final_data')
    parser.add_argument("--save_dir", type=str, default='output')
    parser.add_argument("--num_processes", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    args = parse_args()

    dset_list = [line.strip() for line in open(args.dataset_path, 'r').readlines()]

    for dset_path in tqdm(dset_list, total=len(dset_list)):
        generator_mod = import_module_from_file(args.config)
        generator_cfgs = []

        data_queue = manager.Queue()
        
        full_dset_path = Path(args.root_dir) / dset_path
        save_path: Path = Path(args.root_dir) / args.save_dir / '/'.join(dset_path.split('/')[1:])
        if not save_path.exists():
            save_path.mkdir(parents=True)
        else:
            logger.info('Path existed, skipping')
            continue
        
        for attr in generator_mod.__all__:
            generator_cfg = getattr(generator_mod, attr)([full_dset_path])
            if isinstance(generator_cfg, list):
                generator_cfgs += generator_cfg
            else:
                generator_cfgs.append(generator_cfg)

        for generator_cfg in generator_cfgs:
            db_writer_process = DBWriterProcess(
                data_queue, save_path, generator_cfg
            )
            db_writer_process.start()

            if args.num_processes == 0:
                process_setup(generator_cfg.render_cfg)
                generate_img(data_queue)
                data_queue.put(STOP_TOKEN)
                db_writer_process.join()
            else:
                with mp.Pool(
                    processes=args.num_processes,
                    initializer=process_setup,
                    initargs=(generator_cfg.render_cfg,),
                ) as pool:
                    pool.apply_async(generate_img, args=(data_queue,))
                    pool.close()
                    pool.join()

                data_queue.put(STOP_TOKEN)
                db_writer_process.join()

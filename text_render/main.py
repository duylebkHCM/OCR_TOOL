import argparse
import multiprocessing as mp
import os
import time
from multiprocessing.context import Process
from pathlib import Path
import cv2
import numpy as np

from loguru import logger

from text_renderer.config import import_module_from_file, GenerateCfg
from text_renderer.render import Render

cv2.setNumThreads(1)

STOP_TOKEN = "kill"

# each child process will initialize Render in process_setup
render: Render


class WriterProcess(Process):
    def __init__(
        self,
        data_queue,
        save_dir: Path,
        jpg_quality: int = 95
    ):
        super().__init__()
        self.save_dir = save_dir
        self.jpg_quality = jpg_quality
        self.data_queue = data_queue

    def img_write(self, name: str, image: np.ndarray):
        img_path = os.path.join(self.save_dir, name + ".jpg")
        cv2.imwrite(img_path, image, self.encode_param())
    
    def encode_param(self):
        return [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
    
    def run(self):
        while True:
            m = self.data_queue.get()
            if m == STOP_TOKEN:
                logger.info("DBWriterProcess receive stop token")
                break           
            
            try:
                save_path = self.save_dir.joinpath(m["save_name"])
                self.img_write(save_path.as_posix(), m["image"])                    
                with open(save_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:                    
                    f.write(m["label"])    
                logger.info(f"Successfully save {m['save_name']} sample to disk")             
            except Exception as e:
                logger.exception("DBWriterProcess error")
                raise e


def process_setup(*args):
    global render
    import numpy as np

    # Make sure different process has different random seed
    np.random.seed()

    render = Render(args[0])
    logger.info(f"Finish setup image generate process: {os.getpid()}")
    

def generate_img(data_queue, idx, save_name):
    start_time=time.time()
    data = render(idx)
    logger.info(f"Sample {save_name} complete render after {time.time()-start_time} s")
    if data_queue is not None:
        data_queue.put({"image": data[0], "label": data[1], "save_name":save_name})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="python config file path")
    parser.add_argument("--effects", nargs="+", default=None, help="Choose which effects in the config that will be used")
    parser.add_argument("--buffer", default=1000, help="Maximum number of sample will by save into a sub output directory")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_processes", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    data_queue = manager.Queue()
    args = parse_args()

    generator_mod = import_module_from_file(args.config)
    
    generator_cfgs = []
    for attr in generator_mod.__all__:
        generator_cfg = getattr(generator_mod, attr)()
        if isinstance(generator_cfg, list):
            for cfg in generator_cfg:
                if cfg.cfg_name in args.effects:
                    generator_cfgs.append(cfg)
        else:
            if generator_cfg.cfg_name in args.effects:
                generator_cfgs.append(generator_cfg)
    
    generator_cfg: GenerateCfg
    for generator_cfg in generator_cfgs:
        cfg_name = generator_cfg.cfg_name
        corpuses = generator_cfg.render_cfg.corpus
        if isinstance(corpuses, list) and len(corpuses) > 1:
            main_corpus = list(filter(lambda c: c.cfg.main_text, corpuses))[0]
            num_samples = len(main_corpus.texts)
        else:
            num_samples = len(corpuses.texts)
        
        save_dir = Path(args.root_dir) / args.save_dir / cfg_name
        save_dir:Path
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        
        writer = WriterProcess(data_queue, save_dir=save_dir)
        writer.start()

        if args.num_processes == 0:
            process_setup(generator_cfg.render_cfg)
            for idx in range(num_samples):
                group=int(idx/args.buffer)
                buffer_dir = save_dir.joinpath(str(group))
                if not buffer_dir.exists():
                    buffer_dir.mkdir(parents=True)
                save_name = str(idx).zfill(len(str(num_samples)))
                save_name = buffer_dir.joinpath(save_name)
                generate_img(data_queue, idx, save_name=save_name)
            data_queue.put(STOP_TOKEN)
            writer.join()
        else:
            #TODO complete multiprocess pipeline
            with mp.Pool(
                processes=args.num_processes,
                initializer=process_setup,
                initargs=(generator_cfg.render_cfg,),
            ) as pool:
                for idx in range(num_samples):
                    save_name=str(idx).zfill(len(str(num_samples)))
                    pool.apply_async(generate_img, args=(data_queue, idx, save_name))
                pool.close()
                pool.join()

            data_queue.put(STOP_TOKEN)
            writer.join()

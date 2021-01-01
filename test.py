import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images,save_images_pix2pix
from itertools import islice
from util import html

import cv2
from pathlib import Path
import shutil
import os
def delete_ipynb_checkpoints():
    # delete all .ipynb_checkpoints dir
    for filename in Path(os.getcwd()).glob('**/*.ipynb_checkpoints'):
        try:
            shutil.rmtree(filename)
        except OSError as e:
            print(e)
        else: 
            print("The %s is deleted successfully" % (filename))

delete_ipynb_checkpoints()

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# test stage
for i, data in enumerate(islice(dataset, opt.num_test),start=1):
    model.set_input(data)
    basename = os.path.basename(model.image_paths[0])
    imgname = '%s'%(os.path.splitext(basename)[0])
    print('process input image(%s) %3.3d/%3.3d' % (basename, i, opt.num_test))

    real_A, fake_B, real_B = model.test()           # run inference
    names = ['real_A', 'real_B', 'fake_B']
    images = [real_A, real_B, fake_B]
   
    img_path = 'input_%s' % (imgname)
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)
webpage.save()
import pdb
import src
import glob
import importlib
import os
import cv2

all_submissions = glob.glob('./src/*')


path = 'Images/I1/'
os.makedirs('./results/', exist_ok=True)
for idx,algo in enumerate(all_submissions):
    print('****************\tRunning Awesome Stitcher developed by: {}  | {} of {}\t********************'.format(algo.split(os.sep)[-1],idx,len(all_submissions)))
    try:
        module_name = '{}_{}'.format(algo.split(os.sep)[-1],'stitcher')
        filepath = '{}{}stitcher.py'.format( algo,os.sep,'stitcher.py')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PanaromaStitcher = getattr(module, 'PanaromaStitcher')
        inst = PanaromaStitcher()

        ###
        stitched_image, homography_matrix_list = inst.make_panaroma_for_images_in(path=path)
        cv2.imwrite('./results/{}.png'.format(spec.name),stitched_image)
        print('Panaroma be saved ... @ ./results/{}.png'.format(spec.name))
        print('\n\n')

    except Exception as e:
        print('Oh No! My implementation encountered this issue\n\t{}'.format(e))
        print('\n\n')

import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class PartImageNet(CustomDataset):
    """PartImageNet dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
        
    CLASSES=('Quadruped Head', 'Quadruped Body', 'Quadruped Foot', 'Quadruped Tail', 
             'Biped Head', 'Biped Body', 'Biped Hand', 'Biped Foot', 'Biped Tail', 
             'Fish Head', 'Fish Body', 'Fish Fin', 'Fish Tail', 
             'Bird Head', 'Bird Body', 'Bird Wing', 'Bird Foot', 'Bird Tail', 
             'Snake Head', 'Snake Body', 
             'Reptile Head', 'Reptile Body', 'Reptile Foot', 'Reptile Tail', 
             'Car Body', 'Car Tire', 'Car Side Mirror', 
             'Bicycle Body', 'Bicycle Head', 'Bicycle Seat', 'Bicycle Tire', 
             'Boat Body', 'Boat Sail', 
             'Aeroplane Head', 'Aeroplane Body', 'Aeroplane Engine', 'Aeroplane Wing', 'Aeroplane Tail', 
             'Bottle Mouth', 'Bottle Body')
        
    PALETTE = [ [ 54,  20,  67], [ 77, 111,  75], [115,   0,  51], [ 84, 124, 148], [212,  28, 207], [ 55, 146, 221],
                [ 90, 131, 133], [ 97, 246, 160], [249, 214,  61], [127,  25, 117], [ 45, 188,  43], [ 98, 232, 148],
                [221, 208, 103], [195,  27,  78], [236, 135, 245], [100,  32, 143], [ 66, 140, 139], [149, 109, 192],
                [212,  68, 214], [195,  79, 223], [ 45,   1, 149], [150,  49,  66], [  2,  30,  52], [110, 153, 197],
                [ 51, 159, 252], [126, 160, 125], [125, 112, 209], [201, 165, 102], [143, 144,  50], [ 38, 149, 104],
                [213, 176, 116], [ 19, 171,   5], [184, 112,  29], [106, 237, 110], [248, 153,  58], [ 70,  27, 217],
                [ 21, 165, 227], [186, 230,  53], [ 42, 115, 168], [131, 114,  62]]
        

    def __init__(self, **kwargs) -> None:
        super(PartImageNet, self).__init__(
            img_suffix='.JPEG',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            ignore_index=40,
            **kwargs)
    
    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files

@DATASETS.register_module()    
class PartImageNetCar(CustomDataset):
    """PartImageNet dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES=('Bicycle Body', 'Bicycle Head', 'Bicycle Seat', 'Bicycle Tire')
        
    PALETTE = [[ 54,  20,  67], [ 77, 111,  75], [115,   0,  51], [ 84, 124, 148]]
        

    def __init__(self, **kwargs) -> None:
        super(PartImageNetCar, self).__init__(
            img_suffix='.JPEG',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            ignore_index=3,
            **kwargs)
        
@DATASETS.register_module()    
class PartImageNetBicycle(CustomDataset):
    """PartImageNet dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES=('Bicycle Body', 'Bicycle Head', 'Bicycle Seat', 'Bicycle Tire')
        
    PALETTE = [[ 54,  20,  67], [ 77, 111,  75], [115,   0,  51], [ 84, 124, 148]]
        

    def __init__(self, **kwargs) -> None:
        super(PartImageNetBicycle, self).__init__(
            img_suffix='.JPEG',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            ignore_index=4,
            **kwargs)
    
    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files

@DATASETS.register_module()    
class TDFMBicycle(CustomDataset):
    """PartImageNet dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES=("Bicyle Frame", "Bicyle Wheel", "Bicyle Handlebar", "Bicycle Seat", "Bicycle Pedal", "Bicycle Chain")
    PALETTE = [[ 54,  20,  67], [ 77, 111,  75], [115,   0,  51], [ 84, 124, 148], [212,  28, 207], [ 55, 146, 221]]
        

    def __init__(self, **kwargs) -> None:
        super(TDFMBicycle, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)


@DATASETS.register_module()    
class TDFM_PIN(CustomDataset):
    """3DFroMLLM based dataset made from PartImageNet super categories.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES= ['Biped Head', 'Biped Torso', 'Biped LeftArm', 'Biped RightArm', 'Biped LeftLeg', 'Biped RightLeg', 'Biped LeftFoot', 'Biped RightFoot', 'Quadruped Head', 'Quadruped Snout', 'Quadruped Eye', 'Quadruped Body', 'Quadruped Leg', 'Quadruped Tail', 'Quadruped Ear', 'Snake Head', 'Snake BodySegment', 'Snake Tail', 'Snake Eye', 'Snake Tongue', 'Car Wheel', 'Car CarBody', 'Car CarRoof', 'Car FrontWindow', 'Car RearWindow', 'Car Seat', 'Boat Hull', 'Boat Deck', 'Boat Railing', 'Boat Mast', 'Boat Sail', 'Boat Rudder', 'Bottle Base', 'Bottle Body', 'Bottle Neck', 'Bottle Cap', 'Reptile Body', 'Reptile Head', 'Reptile FrontLeg', 'Reptile BackLeg', 'Reptile Tail', 'Reptile Eye', 'Reptile Tongue', 'Bicycle FrameTopTube', 'Bicycle FrameSeatTube', 'Bicycle FrameDownTube', 'Bicycle FrontWheel', 'Bicycle RearWheel', 'Bicycle Handlebar', 'Bicycle Seat', 'Bicycle LeftPedal', 'Bicycle RightPedal', 'Fish Body', 'Fish Head', 'Fish TailFin', 'Fish DorsalFin', 'Fish PectoralFin', 'Fish Eye', 'Bird Body', 'Aeroplane Fuselage', 'Aeroplane Wing', 'Aeroplane VerticalStabilizer', 'Aeroplane HorizontalStabilizer', 'Aeroplane Engine', 'Aeroplane LandingGear', 'Aeroplane Nose', 'Aeroplane Cockpit', 'Background']
    
    PALETTE = [[224,  99,  23], [ 22,  82, 200], [ 27,  82,  48], [168, 108, 143], [ 26,  88, 239], [ 81,  43,  81], 
               [ 74, 233,  82], [ 86,  57, 136], [ 86, 140, 235], [223, 168, 219], [224,  17, 145], [ 16, 175, 151], 
               [ 95,  13,  48], [132, 152, 136], [187, 165, 180], [151,  84, 150], [228, 205,  27], [252, 143,  79], 
               [ 86,  58, 107], [166, 155, 195], [ 26,  94, 105], [118, 219, 234], [ 29, 223, 182], [163,  58,  80], 
               [133, 114,  36], [159,  51,  13], [179, 221, 220], [ 20,  98,  20], [ 43, 160,  45], [ 82, 164,   7], 
               [106, 137, 124], [ 36, 239, 189], [199, 122, 156], [251,  12,  18], [ 28,  35, 139], [  7, 156, 194], 
               [ 97, 179, 241], [ 66, 150,  28], [  8, 105,  33], [  7, 215, 135], [202, 115, 175], [160,  35,   2], 
               [ 60, 194, 207], [136,  27, 176], [117, 220, 239], [ 35, 190, 227], [ 21, 239,  44], [ 86,  61, 207], 
               [ 64, 131, 111], [ 96, 100,  16], [ 44, 157, 246], [218, 107, 127], [242,  32,  11], [ 45,  19, 149], 
               [160, 245, 170], [ 62,  31, 157], [136,  45,  46], [ 55, 105,  42], [ 84, 216,  57], [166, 169, 179], 
               [ 82, 135,  70], [ 23,   8, 118], [ 69, 149,  65], [183,  46, 200], [223,  86, 188], [199, 243,  54], 
               [ 30, 219,  39], [255, 255, 255]]
        

    def __init__(self, **kwargs) -> None:
        super(TDFM_PIN, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)



# [('Quadruped Head', 0), ('Quadruped Body', 1), ('Quadruped Foot', 2), ('Quadruped Tail', 3), ('Biped Head', 4), ('Biped Body', 5), ('Biped Hand', 6), ('Biped Foot', 7), ('Biped Tail', 8), ('Fish Head', 9), ('Fish Body', 10), ('Fish Fin', 11), ('Fish Tail', 12), ('Bird Head', 13), ('Bird Body', 14), ('Bird Wing', 15), ('Bird Foot', 16), ('Bird Tail', 17), ('Snake Head', 18), ('Snake Body', 19), ('Reptile Head', 20), ('Reptile Body', 21), ('Reptile Foot', 22), ('Reptile Tail', 23), ('Car Body', 24), ('Car Tier', 25), ('Car Side Mirror', 26), ('Bicycle Body', 27), ('Bicycle Head', 28), ('Bicycle Seat', 29), ('Bicycle Tier', 30), ('Boat Body', 31), ('Boat Sail', 32), ('Aeroplane Head', 33), ('Aeroplane Body', 34), ('Aeroplane Engine', 35), ('Aeroplane Wing', 36), ('Aeroplane Tail', 37), ('Bottle Mouth', 38), ('Bottle Body', 39)]
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmenters.weather import CloudLayer
ia.seed(1)


class BBoxFileTranslator:
    def _bbox_yo_yolo_line(self, bbx: BoundingBox, width: int, height: int):
        x_center_n = float(bbx.center_x) / width
        y_center_n = float(bbx.center_y) / height
        w_n = float(bbx.width) / width
        h_n = float(bbx.height) / height    
        return f"{bbx.label} {x_center_n:6f} {y_center_n:6f} {w_n:6f} {h_n:6f}\n"


    def _yolo_line_to_bbox(self, line: str, width: int, height: int):
        coords = line.split()
        class_id = int(coords[0])
        x_center_n, y_center_n, w_n, h_n = map(float, coords[1:5])
        x1_n = x_center_n - w_n / 2
        y1_n = y_center_n - h_n / 2
        x2_n = x_center_n + w_n / 2
        y2_n = y_center_n + h_n / 2
        return BoundingBox(
            x1_n * width, 
            y1_n * height, 
            x2_n * width, 
            y2_n * height, 
            label=class_id
        )
    

    def read_bboxes_from_yolo_like_file(self, path_to_file: Path, width: int, height: int, channels=3):
        lines = path_to_file.read_text().splitlines()
        if len(lines) > 0 and lines[0] != '':
            bboxes = [self._yolo_line_to_bbox(line, width, height) for line in lines]
            return BoundingBoxesOnImage(bboxes, shape=(height, width, channels))
        return BoundingBoxesOnImage([], shape=(height, width, channels))
    

    def write_bboxes_to_yolo_like_file(self, path_to_file: Path, boxes_on_image: BoundingBoxesOnImage):
        with open(str(path_to_file), "w") as file:
            for bbx in boxes_on_image.bounding_boxes:
                yolo_line = self._bbox_yo_yolo_line(bbx, boxes_on_image.width, boxes_on_image.height)
                file.write(yolo_line)


class SelectiveAugmenter:
    def __init__(self):
        self.custom_fog = CloudLayer(
            intensity_mean=(220, 245),# white fog (default values)
            intensity_freq_exponent=-2.5, # (small values -> low frequency -> large structures)
            intensity_coarse_scale=(1.8, 2.2),
            alpha_min=(0.8, 0.95),
            alpha_multiplier=(0.25, 0.3),
            alpha_size_px_max=(2, 5),
            alpha_freq_exponent=(-4.0, -3.0),
            sparsity=(0.85, 0.95),
            density_multiplier=(0.5, 0.75),
        )
        self.rain = iaa.Rain(
            drop_size=(0.01, 0.02),
            speed=(0.1, 0.3)
        )
        self.snow_flakes = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))
        
        self.motion_blur = iaa.MotionBlur(k=5, angle=(-45, 45))
        self.color_temperature = iaa.ChangeColorTemperature(kelvin=(4000, 11000)) # blue cold colors
        self.gaussian_noise = iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)) # 0.05*255 -- for car being visible
        self.perspective_transform = iaa.PerspectiveTransform(scale=(0.01, 0.05), keep_size=True)
        self.contrast = iaa.LinearContrast(alpha=(0.7, 1.5))

        self.coarse_dropout = iaa.CoarseDropout(
            p=0.002,
            size_percent=(0.2, 0.3),
            min_size=4, 
        )
        
        self.night_augmenter = iaa.Sequential([
                iaa.Multiply((0.8, 0.5)),
                iaa.AddToBrightness(add=(-30, -10)),
            ])
        
        self.augmentations = {
            "small_objects": iaa.Sequential([
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    fit_output=True),
            ]),
            "fog": iaa.Sequential(
                [
                    self.custom_fog,
                ]),
            "rain_and_coarse_dropout":
                iaa.Sequential([
                    self.rain,
                    self.coarse_dropout
                ]),
                
            "base_augmentations": iaa.Sequential([
                self.contrast,
                self.color_temperature,
                self.perspective_transform,
                self.motion_blur,
                self.gaussian_noise
            ]),
                    
            "dropout": iaa.Dropout(p=0.002),                
        }

    def detect_fog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) > 160 and np.std(gray) < 50


    def augment(self, image, bbs: BoundingBoxesOnImage):
        aug_seq = []
        
        
        aug_seq.append(self.augmentations["small_objects"])
        
        if not self.detect_fog(image):

            rnd_value = np.random.rand()
            if rnd_value < 0.35:
                # snow or / and rain
                snow_rain_rnd = np.random.rand()
                if snow_rain_rnd < .3:
                    aug_seq.append(self.snow_flakes)
                elif snow_rain_rnd < .7:
                    aug_seq.append(self.augmentations["rain_and_coarse_dropout"])
                else:
                    aug_seq.append(self.augmentations["rain_and_coarse_dropout"])
                    aug_seq.append(self.snow_flakes)

            elif rnd_value < 0.7:
                aug_seq.append(self.augmentations["fog"])
            else:
                aug_seq.append(self.augmentations["base_augmentations"]) 
                
        seq = iaa.Sequential(aug_seq)
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        return image_aug, bbs_aug


class YoloAugmenter:
    def __init__(self, augmenter: SelectiveAugmenter):
        self.augmenter = augmenter
        self.bbox_file_translator = BBoxFileTranslator()

    def augment(self, src_images_dir: Path, src_annotations_dir: Path, augmented_images_dir: Path, augmented_annotations_dir: Path, debug=False):
        src_images_stems = {src_image_path.stem for src_image_path in src_images_dir.glob("*.jpg")}
        src_annotations_stems = {src_annotations_path.stem for src_annotations_path in src_annotations_dir.glob("*.txt")}
        unioun_stems = list(src_images_stems & src_annotations_stems)
        
        
        for image_annotation_stem in tqdm(unioun_stems, ncols=100):
            augmented_image_file = augmented_images_dir.joinpath("augmented_"+image_annotation_stem + ".jpg")
            augmented_annotations_file = augmented_annotations_dir.joinpath("augmented_"+image_annotation_stem + ".txt")

            image_path = src_images_dir.joinpath(image_annotation_stem+".jpg")
            annotations_path = src_annotations_dir.joinpath(image_annotation_stem+".txt")
            image = cv2.imread(str(image_path))
            
            h, w, _ = image.shape
            bboxes = self.bbox_file_translator.read_bboxes_from_yolo_like_file(annotations_path, w, h)
            augmented_image, augmented_bboxes = self.augmenter.augment(image, bboxes)
            if debug:
                augmented_image = augmented_bboxes.draw_on_image(augmented_image)
                tqdm.write(f"image: {image_annotation_stem}, image shape: {image.shape}| parsed h, w, channels: {h}, {w}, {_}")
                tqdm.write(f"bbox: {bboxes.bounding_boxes[0]}, image size: w = {bboxes.width}, h = {bboxes.height}")
                tqdm.write(f"augmented bbox: {augmented_bboxes.bounding_boxes[0]}, image size: w = {augmented_bboxes.width}, h = {augmented_bboxes.height}")
                tqdm.write("---"*30)
            cv2.imwrite(str(augmented_image_file), augmented_image)
            self.bbox_file_translator.write_bboxes_to_yolo_like_file(augmented_annotations_file, augmented_bboxes)


if __name__ == "__main__":
    root = Path("/home/andrei/cams-set/augmentations/test_aug")
    src_images_dir = root.joinpath("images_test")
    src_annotations_dir = root.joinpath("labels_test")
    augmented_images_dir = root.joinpath("augmented_images")
    augmented_annotations_dir = root.joinpath("augmented_annotations")
    print(src_images_dir)
    print(src_annotations_dir)
    print(augmented_images_dir)
    print(augmented_annotations_dir)
    augmenter = SelectiveAugmenter()
    yolo_augmenter = YoloAugmenter(augmenter)
    
    yolo_augmenter.augment(src_images_dir, src_annotations_dir, augmented_images_dir, augmented_annotations_dir, debug=True)
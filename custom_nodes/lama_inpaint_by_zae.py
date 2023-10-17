import math
import cv2
import numpy as np
import torch
import image_ocr


from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, SDSampler
from simple_lama_inpainting import SimpleLama


class OCR:
    __thick = 2
    __pipe = None
    __px = 255  # For 1-channel image
    __kernel = np.ones((3, 3), dtype=np.uint8)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "masking"

    # OUTPUT_NODE = False

    CATEGORY = "MP-custom2"

    @classmethod
    def rect_fill(cls, mask, pts):
        lt = np.array(pts[0], dtype=int)
        rb = np.array(pts[2], dtype=int)
        mask = cv2.rectangle(mask, lt, rb, cls.__px, -1)
        mask = cv2.dilate(mask, kernel=cls.__kernel, iterations=2)
        return mask

    @classmethod
    def poly_fill(cls, mask, pts):
        return cv2.fillPoly(mask, np.array([pts], dtype=int), cls.__px)

    @classmethod
    def masking(cls, image, flag: str = "line"):
        img = np.array(image[0] * 255, dtype=np.uint8)
        pipe = cls.get_pipe()
        try:
            results = pipe.recognize([img])[0]
        except IndexError:
            # Cannot detect text in given image
            return (None, )

        mask = np.zeros(img.shape[:2], dtype="uint8")
        for txt, box in results:
            if flag == "line":
                [x0, y0] = box[0]  # LT
                [x1, y1] = box[1]  # RT
                [x2, y2] = box[2]  # RB
                [x3, y3] = box[3]  # LB

                x_mid0, y_mid0 = int((x1 + x2) // 2), int((y1 + y2) // 2)
                x_mid1, y_mid1 = int((x0 + x3) // 2), int((y0 + y3) // 2)

                thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.3)

                # Define the line and inpaint
                mask = cv2.line(
                    mask, (x_mid0, y_mid0), (x_mid1, y_mid1), cls.__px, thickness
                )
            elif flag == "rectangle":
                mask = cls.rect_fill(mask, box)
            elif flag == "polygon":
                mask = cls.poly_fill(mask, box)
        return (torch.tensor(mask / 255), )

    @classmethod
    def set_pipe(cls):
        cls.__pipe = image_ocr.pipeline.Pipeline()  # OCR pipeline

    @classmethod
    def get_pipe(cls):
        if OCR.__pipe is None:
            cls.set_pipe()
        return cls.__pipe


class LamaCleaner:
    def __init__(self):
        self.model = None
        self.cfg = None
        self.get_model()
        self.get_cfg()

    def get_model(self):
        model = ModelManager(
            name="lama",
            sd_controlnet=True,
            sd_controlnet_method="sontrol_v11p_sd15_canny",
            device=torch.device("cuda"),
            no_half=True,
            hf_access_token="",
            disable_nsfw=True,
            sd_cpu_textencoder=True,
            sd_run_local=True,
            sd_local_model_path="",
            local_files_only=True,
            cpu_offload=True,
            enable_xformers=True,
        )
        self.model = model

    def get_cfg(self):
        cfg = Config(
            ldm_steps=10,
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=128,
            hd_strategy_crop_trigger_size=128,
            hd_strategy_resize_limit=128,
            prompt="",
            sd_steps=20,
            sd_sampler=SDSampler.ddim
        )
        self.cfg = cfg

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "run"

    # OUTPUT_NODE = False

    CATEGORY = "MP-custom2"

    def run(self, image, mask):
        image = np.array(image)
        mask = np.array(mask)

        if image.shape[1:3] != mask.shape[-2:]:
            print(f"Expect given arguments have same shape, But image has {image.shape} and mask has {mask.shape}.")
            print(f"Resize mask to image shape. e.g. {mask.shape[1:3]} to {image.shape[-2:]}.")
            self.resize_mode = True

        # image = torch.permute(image, [0, 3, 1, 2])

        lamas = []
        for img, m in zip(image, mask):
            m = m.squeeze()
            if self.resize_mode:
                m = cv2.resize(m, dsize=img.shape[1::-1])
            lamas.append(self.model(img, m, self.cfg)[:, :, ::-1])

        return (torch.tensor(np.stack(lamas, axis=0)), )


class LamaInpaint:
    """
    Inpainting images custom node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        self.simple_lama = SimpleLama()
        self.resize_mode = False

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "inpaint_lama"

    # OUTPUT_NODE = False

    CATEGORY = "MP-custom2"

    def inpaint_lama(self, image, mask):
        print(f'{image.min()} - {image.max()}: {image.shape} w/ dtype {image.dtype}')
        print(f'{mask.min()} - {mask.max()}: {mask.shape} w/ dtype {mask.dtype}')

        image = np.array(image * 255, dtype=np.uint8)
        mask = np.array(mask * 255, dtype=np.uint8)

        # mask = torch.permute(mask, [0, 2, 3, 1]) * 255
        print(f'{image.min()} - {image.max()}: {image.shape} w/ dtype {image.dtype}')
        print(f'{mask.min()} - {mask.max()}: {mask.shape} w/ dtype {mask.dtype}')

        if image.shape != mask.shape:
            print(f"Expect given arguments have same shape, But image has {image.shape} and mask has {mask.shape}.")
            print(f"Resize mask to image shape. e.g. {mask.shape} to {image.shape[1:3]}.")
            self.resize_mode = True

        lamas = []
        for img in image:
            m = cv2.resize(mask, dsize=img.shape[1::-1]) if self.resize_mode else mask
            print(f'mask shape is {m.shape}')
            lama = np.array(self.simple_lama(img, m))
            if lama.shape != img.shape:
                lama = cv2.resize(lama, dsize=img.shape[1::-1])
            lamas.append(torch.tensor(lama))
        result = torch.stack(lamas, dim=0)
        print(f'return {result.shape} tensor, range [{result.min()} - {result.max()}]')

        return (result / 255, )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Lama Inpaint": LamaInpaint,
    "Lama Cleaner": LamaCleaner,
    "Image OCR": OCR
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Lama inpainting by urf"
}

#
if __name__ == "__main__":
    img = np.random.random((1, 884, 600, 3))
    mask = np.random.random((1125, 764))

    lc = LamaCleaner()
    lama = LamaInpaint()

    res = lama.inpaint_lama(img, mask)
    res2 = lc.run(img, mask)
    print(1)

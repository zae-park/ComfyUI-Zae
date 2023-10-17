import cv2
import torch
from torchvision.transforms import Resize
from comfy_extras.nodes_post_processing import Blend


class ImageAdd:
    """
    Add images custom node

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
        pass

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
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "imadd"

    # OUTPUT_NODE = False

    CATEGORY = "MP-custom"

    # def test(self, image, string_field, int_field, float_field, print_to_screen):
    #     if print_to_screen == "enable":
    #         print(f"""Your input contains:
    #             string_field aka input text: {string_field}
    #             int_field: {int_field}
    #             float_field: {float_field}
    #         """)
    #     # do some processing on the image, in this example I just invert it
    #     image = 1.0 - image
    #     return (image,)

    def imadd(self, image1, image2):
        """
        Add two images.
        Output image is weighted sum of given 2 images using mask of 2nd image.
        :param img1: Image.
        :param img2: Image.
        :return: Image.
        """
        img1 = torch.permute(image1, [0, 3, 1, 2])      # [B, H, W, C] to [B, C, H, W]
        img2 = torch.permute(image2, [0, 3, 1, 2])
        if image1.shape != image2.shape:
            print(f"Expect given 2 images have same shape, But 1st image has {image1.shape}, but 2nd img has .")
            print(f"Resize 2nd img to 1st img shape. e.g. {image2.shape} to {image1.shape}.")
            resizer = Resize(img1.shape[2:])
            img2 = resizer(img2)

        mask = self.masking(img2) * 1
        result = img1 * (1 - mask) + img2 * mask
        result = torch.permute(result, [0, 2, 3, 1])
        print(image1.shape)
        print(result.shape)
        return (result,)

    def masking(self, img):
        proj = torch.sum(img, dim=1, keepdim=True)
        mask = proj != 0
        return mask


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image addition": ImageAdd
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Image Add by urf"
}


if __name__ == "__main__":
    tmp1 = torch.zeros((1, 884, 600, 3))
    tmp2 = torch.zeros((1, 1381, 938, 3))
    imadd = ImageAdd()
    res = imadd.imadd(tmp1, tmp2)
    print(1)

from typing import Tuple

import torchvision
from torch import Tensor

import global_vars


class ImageHandler:

    @staticmethod
    def rescale_to_zero_one(images: Tensor):
        assert (images >= 0).all(), "it's expected that all values are non-negative."
        amplified_images: Tensor = images / images.max()
        return amplified_images

    @staticmethod
    def add_input_images(input_images: Tensor, caption: str = "some input images"):
        # input_images = input_images * ImageHandler.STD_DEV_MNIST + ImageHandler.MEAN_MNIST  # un-normalize
        input_images = ImageHandler.un_normalize(input_images)  # un-normalize
        print('Adding input images.')
        # global step is zero, as this is only done once.
        ImageHandler.add_image_grid_to_writer(caption, input_images)

    @staticmethod
    def add_gradient_images(test_batch: Tuple[Tensor, Tensor], learner, additional_caption: str,
                            global_step: int = None, save=False):
        test_images, test_labels = test_batch
        rescaled_explanation_batch: Tensor = learner.get_explanation_batch(test_images, test_labels, save=save)

        # if the combination is multiplication, show only the gradient as well
        if learner.explanation_mode == "input_x_gradient" \
                or learner.explanation_mode == "input_x_integrated_gradient":
            gradient = rescaled_explanation_batch/test_images
            ImageHandler.add_image_grid_to_writer(caption=f"gradient/{additional_caption}",
                                                  some_images=gradient,
                                                  global_step=global_step)
            # un-normalizing is also only necessary if we are in input-space.
            explanation_batch_show = ImageHandler.un_normalize(rescaled_explanation_batch)
        else:
            explanation_batch_show = rescaled_explanation_batch

        ImageHandler.add_image_grid_to_writer(caption=f"{learner.explanation_mode}/{additional_caption}",
                                              some_images=explanation_batch_show,
                                              global_step=global_step)

    @staticmethod
    def add_image_grid_to_writer(caption: str, some_images: Tensor, global_step: int = None):
        combined_image = torchvision.utils.make_grid(some_images)

        if global_vars.LOGGING:
            global_vars.LOGGING.writer.add_image(caption, combined_image, global_step=global_step)
        else:
            print(colored(200, 0, 0, f"No writer set - skipped adding {caption} images."))

    @staticmethod
    def un_normalize(images):
        return images.mul_(global_vars.STD).add_(global_vars.MEAN)


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

"""
RandAugment pools.

The 14 default augmentations used in RandAugment with the range of their parameter.
If no parameter is available, then the range is None.

- An augment is range must have a constructor with 1 parameter.
- An augment without range must have a constructor without any parameters.
"""

from sslh.transforms.image.pil import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    IdentityImage,
    Invert,
    Posterize,
    Rotate,
    Sharpness,
    ShearX,
    ShearY,
    Solarize,
    TranslateX,
    TranslateY,
)

RAND_AUGMENT_POOL_1 = [
    (AutoContrast, None),
    (Brightness, (-0.9, 0.9)),
    (Color, (-0.9, 0.9)),
    (Contrast, (-0.9, 0.9)),
    (Equalize, None),
    (IdentityImage, None),
    (Posterize, (0, 4)),
    (Rotate, (-30, 30)),
    (Sharpness, (-0.9, 0.9)),
    (ShearX, (-0.3, 0.3)),
    (ShearY, (-0.3, 0.3)),
    (Solarize, (0, 256)),
    (TranslateX, (-0.3, 0.3)),
    (TranslateY, (-0.3, 0.3)),
]
RAND_AUGMENT_POOL_2 = [
    (AutoContrast, None),
    (Brightness, (-0.9, 0.9)),
    (Color, (-0.9, 0.9)),
    (Contrast, (-0.9, 0.9)),
    (Equalize, None),
    (Invert, None),
    (Posterize, (0, 4)),
    (Rotate, (-30, 30)),
    (Sharpness, (-0.9, 0.9)),
    (ShearX, (-0.3, 0.3)),
    (ShearY, (-0.3, 0.3)),
    (Solarize, (0, 256)),
    (TranslateX, (-0.3, 0.3)),
    (TranslateY, (-0.3, 0.3)),
]

RAND_AUGMENT_DEFAULT_POOL = RAND_AUGMENT_POOL_1

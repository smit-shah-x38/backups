from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl
import torch


# Define your data model
class MyDocument(BaseDoc):
    description: str
    image_url: ImageUrl  # could also be VideoUrl, AudioUrl, etc.
    image_tensor: TorchTensor[1704, 2272, 3]  # you can express tensor shapes!


# Stack multiple documents in a Document Vector
from docarray import DocVec

vec = DocVec[MyDocument](
    [
        MyDocument(
            description="A cat",
            image_url="https://example.com/cat.jpg",
            image_tensor=torch.rand(1704, 2272, 3),
        ),
    ]
    * 10
)
print(vec.image_tensor.shape)  # (10, 1704, 2272, 3)

from typing import Sequence, MutableSequence, Callable, Iterator
from nptyping import NDArray, Shape, Float
from pygame import SurfaceType
from pygame.event import Event

Vec3 = NDArray[Shape["3"], Float] | list[float]
VectorList = NDArray[Shape["*,3"], Float] | list[Vec3]
ShapeList = NDArray[Shape["*,*,3"], Float] | NDArray[Shape["*,*,*,3"], Float]
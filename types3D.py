from typing import Sequence, MutableSequence, Callable, Any, Iterable

from nptyping import NDArray, Shape, Int, Float, Bool

from pygame import SurfaceType
from pygame.event import Event


BoolArray = NDArray[Any, Bool]
Vec3 = NDArray[Shape["3"], Float] | list[float]


# Either a linear list of vectors, or one vector per face in a shape (e.g. normals).
VectorArray = (
	NDArray[Shape["*,3"], Float] |
	NDArray[Shape["*,*,3"], Float]
)

# Either a single shape (of n faces, each with m corners), or a list of such shapes.
ShapeArray = (
	NDArray[Shape["*,*,3"], Float] |
	NDArray[Shape["*,*,*,3"], Float]
)

# List of polygons with some number of corners, each with 2 screen coordinates.
PolygonArray = NDArray[Shape["*,*,2"], Float]

# Alpha channel is optional.
Color = (
	Sequence[int|float] |
	NDArray[Shape["3"], Float] |
	NDArray[Shape["4"], Float]
)
ColorArray = (
	NDArray[Shape["*,3"], Float] |
	NDArray[Shape["*,4"], Float]
)

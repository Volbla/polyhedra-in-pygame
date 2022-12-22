from types3D import *

import numpy as np

def multi_span_boxes(min_corners:Vec3|Iterator, max_corners:Vec3|Iterator|None=None) -> ShapeList:
	if max_corners is not None:
		return np.array([span_box(*corners) for corners in zip(min_corners, max_corners)])
	else:
		return np.array([span_box(corners) for corners in min_corners])

def multi_skeleton_boxes(min_corners:Vec3|Iterator, max_corners:Vec3|Iterator|None=None) -> ShapeList:
	if max_corners is not None:
		return np.array([skeleton_box(*corners) for corners in zip(min_corners, max_corners)])
	else:
		return np.array([skeleton_box(corners) for corners in min_corners])


def span_box(min_corner:Vec3, max_corner:Vec3|None=None) -> NDArray[Shape["6,5,3"], Float]:
	"""Takes two points as opposite corners of an axis-aligned box.
	Returns 6 faces, each with 4 corners and 1 normal.
	"""

	if max_corner is None:
		max_corner = np.array(min_corner) + 1.0

	cubular = np.array([
		[[0,0,0], [0,1,0], [1,1,0], [1,0,0]],
		[[0,0,0], [1,0,0], [1,0,1], [0,0,1]],
		[[0,0,0], [0,0,1], [0,1,1], [0,1,0]],
		[[1,1,1], [0,1,1], [0,0,1], [1,0,1]],
		[[1,1,1], [1,1,0], [0,1,0], [0,1,1]],
		[[1,1,1], [1,0,1], [1,0,0], [1,1,0]],
	])
	squares = np.where(cubular, max_corner, min_corner)
	normals = normal(squares)

	return np.concatenate((squares, normals[:, np.newaxis, :]), axis=1)

def skeleton_box(min_corner:Vec3, max_corner:Vec3|None=None) -> VectorList:
	"""Generates the same boxes as span_box, but only returns
	the 8 corners of the cube. For faster filtering calculations.
	"""

	if max_corner is None:
		max_corner = np.array(min_corner) + 1.0

	cubular = np.array([
		[0,0,0], [0,1,0], [1,1,0], [1,0,0],
		[1,1,1], [1,0,0], [1,0,1], [0,0,1],
	])
	corners = np.where(cubular, max_corner, min_corner)
	return corners

def normal(planes:ShapeList) -> VectorList:
	"""Normal of a flat surface of at least three points.
	Direction is right-handed/anti-clockwise to the order of the points.
	Accepts arrays of multiple planes.
	"""

	vectors = planes[..., [1,-1], :] - planes[..., [0,0], :]
	normal = np.cross(vectors[..., 0, :], vectors[..., 1, :])
	# Normalizing
	length = np.sqrt(np.einsum("...i, ...i", normal, normal))
	normnorm = normal / length[:,np.newaxis]
	return normnorm

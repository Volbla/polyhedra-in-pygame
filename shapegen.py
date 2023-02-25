from types3D import *
from transformations import arraydot

import numpy as np
from numpy import array


def multi_span_boxes(min_corners:Iterable[Vec3], max_corners:Iterable[Vec3]|None=None) -> ShapeArray:
	if max_corners is not None:
		return array([span_box(*corners) for corners in zip(min_corners, max_corners)])
	else:
		return array([span_box(corners) for corners in min_corners])


def multi_skeleton_boxes(min_corners:Iterable[Vec3], max_corners:Iterable[Vec3]|None=None) -> ShapeArray:
	if max_corners is not None:
		return array([skeleton_box(*corners) for corners in zip(min_corners, max_corners)])
	else:
		return array([skeleton_box(corners) for corners in min_corners])


def span_box(min_corner:Vec3, max_corner:Vec3|None=None) -> ShapeArray:
	"""Takes two points as opposite corners of an axis-aligned box.

	If max_corner is not specified, defaults to a cube with side = 1.
	Returns 6 faces, each with 4 corners and 1 normal.
	"""

	if max_corner is None:
		max_corner = array(min_corner) + 1.0

	cubular = array([
		[[0,0,0], [0,1,0], [1,1,0], [1,0,0]],
		[[0,0,0], [1,0,0], [1,0,1], [0,0,1]],
		[[0,0,0], [0,0,1], [0,1,1], [0,1,0]],
		[[1,1,1], [0,1,1], [0,0,1], [1,0,1]],
		[[1,1,1], [1,1,0], [0,1,0], [0,1,1]],
		[[1,1,1], [1,0,1], [1,0,0], [1,1,0]],
	])
	squares = np.where(cubular, max_corner, min_corner)
	normals = normal(squares)

	return np.concatenate((squares, normals[:, None, :]), axis=1)


def skeleton_box(min_corner:Vec3, max_corner:Vec3|None=None) -> VectorArray:
	"""Generates the same boxes as span_box, but only returns
	the 8 corners of the cube. For faster filtering calculations.
	"""

	if max_corner is None:
		max_corner = array(min_corner) + 1.0

	cubular = array([
		[0,0,0], [0,1,0], [1,1,0], [1,0,0],
		[1,1,1], [1,0,0], [1,0,1], [0,0,1],
	])
	corners = np.where(cubular, max_corner, min_corner)

	return corners


def normal(planes:ShapeArray) -> VectorArray:
	"""Normal of a flat surface of at least three points.

	Direction is right-handed/anti-clockwise to the order of the points.
	Accepts arrays of multiple planes.
	"""

	vectors = planes[..., [1,2], :] - planes[..., [0,1], :]
	normal = np.cross(vectors[..., 0, :], vectors[..., 1, :])

	# Normalizing
	length = np.sqrt(arraydot(normal, normal))
	normnorm = normal / length[:, None]

	return normnorm

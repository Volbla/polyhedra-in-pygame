from types3D import *

import numpy as np
from numpy import sin, cos
from functools import lru_cache

@lru_cache(maxsize=1)
def rotation_matrix(polar_angles:tuple[float,float]):
	mu, nu = polar_angles
	planmat = np.mat([
			[cos(mu), -sin(mu), 0],
			[sin(mu), cos(mu), 0],
			[0, 0, 1]
	])
	vertmat = np.mat([
			[1, 0, 0],
			[0, cos(nu), sin(nu)],
			[0, -sin(nu), cos(nu)]
	])
	return vertmat @ planmat

def rotate(shapes, polar_angles):
	rotmat = rotation_matrix(tuple(polar_angles))
	return np.einsum("mn, ...n", rotmat, shapes)

def front_face_mask(normals):
	"""For back face culling."""
	return normals[..., 1] < 0

def draw_order_mask(faces):
	"""Draw farther faces first."""
	closest_corner = np.amin(faces[..., 1], axis=1)
	return np.flip(np.argsort(closest_corner))

def project(shapes, zoom, screen_size):
	"""Orthogonal projection."""
	projection = shapes[..., [0,2]] * zoom
	pixels = projection * [1,-1] + np.array(screen_size, dtype=int) // 2

	return pixels

def shade(colors, normals, light_direction):
	"""Mediocre color shading."""
	if not isinstance(colors, NDArray):
		colors = np.array(colors)

	# dot product
	weights = (np.einsum("...n, n", normals, light_direction) + 1) / 2

	channel_count = colors.shape[-1]
	channel_weights = np.tile(weights[...,np.newaxis], channel_count)
	if channel_count == 4:
		# Don't touch the alpha channel
		channel_weights[..., -1] = 1

	return channel_weights * colors[..., np.newaxis, :]
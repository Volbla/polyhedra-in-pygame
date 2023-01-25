from types3D import *
from shapegen import *
from transformations import *

import pygame as pg
import numpy as np
from itertools import product

TAU = 2 * np.pi
SCREEN_SIZE = (1200, 900)
LIGHTDIRR = np.array((1,-0.3,1)) / np.sqrt(2.09)

def main():
	"""Creates the main window, drawable surfaces,
	and coordinates that rotate the camera around the origin.

	3D coordinates are right-handed with z-axis up.

	Calls one available implementation.
	"""

	main_window = start_engine("Explosion", SCREEN_SIZE, (240,)*3)
	opaque = pg.Surface(SCREEN_SIZE)
	transparent = pg.Surface(SCREEN_SIZE, flags=pg.SRCALPHA)
	polar_coordinates:MutableSequence = [30, -TAU/16, -TAU/16]

	cube_of_cubes(main_window, transparent, polar_coordinates)
	# advent(main_window, polar_coordinates)

"""Implementations."""

def cube_of_cubes(window:SurfaceType, canvas:SurfaceType, coords):
	"""Intended to visualize the reach of minecraft explosions."""

	color = (160,200,255)
	big_cube = span_box([-10,-10,-10], [10,10,10])
	small_cubes = multi_span_boxes([a,b,c] for a, b, c in product(range(-10, 10), repeat=3))
	test_cubes = multi_skeleton_boxes([a,b,c] for a, b, c in product(range(-10, 10), repeat=3))

	# These are drawn transparently over the big cube to erase parts of it.
	sides = small_cubes[np.any((small_cubes == 10) | (small_cubes == -10), axis=(1,2,3))]
	test_sides = test_cubes[np.any((test_cubes == 10) | (test_cubes == -10), axis=(1,2))]
	# Must expand slightly to cover the edges.
	sides[(sides == 10) | (sides == -10)] *= 1.003

	far3, close3 = sphere_partition_masks(test_cubes)
	mesh_flags = [True, True, True]
	def render():
		"""Rendering function to be called by the main game loop."""

		angles = coords[1:]
		test_rot = rotate(test_cubes, angles)

		# The small cubes that should be drawn.
		far, close = distance_partition_masks(test_rot)
		# The blank faces to be erased from the big cube.
		far2, close2 = distance_partition_masks(rotate(test_sides, angles))

		canvas.fill((0,0,0,0))
		draw_polygons(canvas, big_cube, color, coords)
		if mesh_flags[0]:
			draw_polygons(canvas, sides[close2 & ~far2], (0,0,0,0), coords)
		if mesh_flags[1]:
			draw_polygons(canvas, small_cubes[far & close3 & far3], color, coords)
		if mesh_flags[2]:
			draw_polygons(canvas, small_cubes[close & far & ~(close3 & ~far3)], color, coords)

		window.fill((240,240,240))
		window.blit(canvas, (0,0))

		pg.display.update()

	render()

	while True:
		event = pg.event.wait()
		if event.type == pg.QUIT or pg.key.get_pressed()[pg.K_ESCAPE]:
			break

		if event.type == pg.KEYDOWN:
			try:
				i = int(event.dict["unicode"]) - 1
			except ValueError:
				continue

			mesh_flags[i] = not mesh_flags[i]
			render()

		if mouse_input(event, coords):
			render()

	pg.quit()

def advent(window:SurfaceType, coords):
	"""Data from Advent of Code 2022 day 18."""

	def center(points):
		"""Shifts a list of unit cube coordinates to center around the origin."""

		ma = np.amax(points, axis=0) + 1
		mi = np.amin(points, axis=0)
		return points - mi - (ma - mi) / 2

	# with open("18.txt", "r", newline="\n") as f:
	# 	inptext = f.read().splitlines()
	# unit_cube_coords = np.array([[float(x) for x in line.split(",")] for line in inptext])
	unit_cube_coords = np.array([[2,2,2],[1,2,2],[3,2,2],[2,1,2],[2,3,2],[2,2,1],[2,2,3],[2,2,4],[2,2,6],[1,2,5],[3,2,5],[2,1,5],[2,3,5]])

	lava_drop = multi_span_boxes(center(unit_cube_coords))
	test_cubes = multi_skeleton_boxes(center(unit_cube_coords))

	# from util.colorwheel import hue_streak
	# colors = np.array(list(hue_streak(len(drop))))
	colors = (150,200,230)

	def render():
		"""Rendering function to be called by the main game loop."""

		far, _ = distance_partition_masks(rotate(test_cubes, coords[1:]))

		window.fill((240,240,240))
		draw_polygons(window, lava_drop[far], colors, coords)
		pg.display.update()

	render()

	while True:
		event = pg.event.wait()
		if event.type == pg.QUIT or pg.key.get_pressed()[pg.K_ESCAPE]:
			break

		if mouse_input(event, coords):
			render()

	pg.quit()

"""Graphics things."""

def draw_polygons(surface:SurfaceType, shapes:ShapeArray, color, coords):
	"""Project polygons in 3D space onto the given surface."""

	zoom, angles = coords[0], coords[1:]

	rotated_shapes = rotate(shapes, angles)
	# Hardcoded 4 corners per face since i only draw squares at the moment.
	faces = rotated_shapes[..., :4, :]
	normals = rotated_shapes[..., 4, :]

	cull = front_face_mask(normals)
	visible = faces[cull]
	shaded_colors = shade(color, normals[cull], LIGHTDIRR)

	order = draw_order_mask(visible)
	pixels = project(visible, zoom, SCREEN_SIZE)

	for points, col in zip(pixels[order], shaded_colors[order]):
		pg.draw.polygon(surface, col, points)

def distance_partition_masks(shapes:ShapeArray, distance:float=0) -> tuple[BoolArray,BoolArray]:
	"""Mask a group of shapes to either side of a plane parallell to the viewing plane.

	Distance is measured from the origin.
	The two masks overlap in the middle.
	Use boolean operations to isolate specific sections.
	"""

	y = shapes[...,1]
	far = np.any(y > distance, axis=-1)
	close = np.any(y < distance, axis=-1)
	return far, close

def sphere_partition_masks(shapes:ShapeArray, center:Vec3=[0,0,0], radius:float=7) -> tuple[BoolArray,BoolArray]:
	relative = shapes - center
	distance2 = arraydot(relative, relative)
	far = np.any(distance2 > radius ** 2, axis=-1)
	close = np.any(distance2 < radius ** 2, axis=-1)
	return far, close

"""Pygame things."""

def mouse_input(event:Event, polar_coordinates:MutableSequence) -> bool:
	"""Updates object rotation and zoom from mouse interactions.

	Returns whether the screen needs to update.
	"""

	if event.type == pg.MOUSEBUTTONUP:
		pg.event.set_blocked(pg.MOUSEMOTION)
		return False

	if event.type == pg.MOUSEBUTTONDOWN:
		pg.event.set_allowed(pg.MOUSEMOTION)
		# Calling get_rel() once so the next call is relative to current position
		pg.mouse.get_rel()
		return False

	if event.type == pg.MOUSEMOTION:
		delta = pg.mouse.get_rel()
		x, y = delta[0], delta[1]

		polar_coordinates[1] += x / (TAU * 20)
		polar_coordinates[1] %= TAU

		polar_coordinates[2] -= y / (TAU * 20)
		polar_coordinates[2] = max(min((polar_coordinates[2]), TAU/4), -TAU/4)
		return True

	if event.type == pg.MOUSEWHEEL:
		polar_coordinates[0] *= 1.15 ** event.y
		return True

	return False

def start_engine(title:str, size:tuple[int,int], color:tuple[int,int,int]) -> SurfaceType:
	pg.init()
	pg.display.set_caption(title)
	window = pg.display.set_mode(size=size)
	window.fill(color)

	# We only need mouse movement when the mouse button is held down.
	pg.event.set_blocked(pg.MOUSEMOTION)
	# Clear any events queued by initialization.
	pg.event.clear()

	return window

if __name__=="__main__":
	print()
	main()

from types3D import *
from shapegen import *
from transformations import *

import pygame as pg
import numpy as np
from itertools import product

TAU = 2 * np.pi
SCREEN_SIZE = (1200, 900)
LIGHTDIRR = np.array((1,-1,1)) / np.sqrt(3)

def main():
	main_window = start_engine("Explosion", SCREEN_SIZE, (240,)*3)
	opaque = pg.Surface(SCREEN_SIZE)
	transparent = pg.Surface(SCREEN_SIZE, flags=pg.SRCALPHA)
	polar_coordinates:MutableSequence = [25, -TAU/16, -TAU/16]

	cube_of_cubes(main_window, transparent, polar_coordinates)
	# advent(main_window, coords)

# Implementations

def cube_of_cubes(window:SurfaceType, canvas:SurfaceType, coords):
	"""Originally made to visuallize the reach of minecraft explosions."""

	color = (150,200,230)
	big_cube = span_box([-10,-10,-10], [10,10,10])
	small_cubes = multi_span_boxes([a,b,c] for a, b, c in product(range(-10, 10), repeat=3))
	test_cubes = multi_skeleton_boxes([a,b,c] for a, b, c in product(range(-10, 10), repeat=3))

	# These are drawn transparently over the big cube to remove parts of it.
	sides = small_cubes[np.any((small_cubes == 10) | (small_cubes == -10), axis=(1,2,3))]
	test_sides = test_cubes[np.any((test_cubes == 10) | (test_cubes == -10), axis=(1,2))]
	sides[(sides == 10) | (sides == -10)] *= 1.003

	def render():
		coords[1] += 0.2
		far, close = distance_partition_masks(rotate(test_cubes, coords[1:]))
		far2, close2 = distance_partition_masks(rotate(test_sides, coords[1:]))

		canvas.fill((0,0,0,0))
		draw_polygons(canvas, big_cube, color, coords)
		draw_polygons(canvas, sides[close2 & ~far2], (0,0,0,0), coords)
		draw_polygons(canvas, small_cubes[close & far], color, coords)

		window.fill((240,240,240))
		window.blit(canvas, (0,0))

		pg.display.update()

	render()
	while running(render, coords):
		pass
	pg.quit()

def advent(window:SurfaceType, coords):
	"""Data from Advent of Code 2022 day 18."""

	def center(points):
		ma = np.amax(points, axis=0) + 1
		mi = np.amin(points, axis=0)
		return points - mi - (ma - mi) / 2

	example = np.array([[2,2,2],[1,2,2],[3,2,2],[2,1,2],[2,3,2],[2,2,1],[2,2,3],[2,2,4],[2,2,6],[1,2,5],[3,2,5],[2,1,5],[2,3,5]])
	with open("18.txt", "r", newline="\n") as f:
		inptext = f.read().splitlines()
	unit_cube_coords = np.array([[float(x) for x in line.split(",")] for line in inptext])

	drop = multi_span_boxes(center(unit_cube_coords))
	test_cubes = multi_skeleton_boxes(center(unit_cube_coords))

	# from util.colorwheel import hue_streak
	# colors = np.array(list(hue_streak(len(drop))))
	colors = (150,200,230)

	def render():
		far, _ = distance_partition_masks(rotate(test_cubes, coords[1:]))

		window.fill((240,240,240))
		draw_polygons(window, drop[far], colors, coords)
		pg.display.update()

	render()
	while running(render, coords):
		pass
	pg.quit()

# Things that things do

def running(renderer:Callable, coords) -> bool:
	event = pg.event.wait()
	if event.type == pg.QUIT or pg.key.get_pressed()[pg.K_ESCAPE]:
		return False
	if mouse_input(event, coords):
		renderer()
	return True

def draw_polygons(surface:SurfaceType, cubes:ShapeList, color, coords):
	zoom, angles = coords[0], coords[1:]

	rotation = rotate(cubes, angles)
	cubes = rotation[..., :4, :]
	normals = rotation[..., 4, :]
	tints = shade(color, normals, LIGHTDIRR)

	cull = front_face_mask(normals)
	visible = cubes[cull]
	pixels = project(visible, zoom, SCREEN_SIZE)
	order = draw_order_mask(visible)

	for points, col in zip(pixels[order], tints[cull][order]):
		pg.draw.polygon(surface, col, points)

def distance_partition_masks(shapes:ShapeList, distance:float=0):
	"""Mask a group of shapes to two sides of a plane.
	The two masks overlap at the plane.
	"""

	y = shapes[...,1]
	far = np.any(y > distance, axis=1)
	close = np.any(y < distance, axis=1)
	return far, close

# Pygame things

def mouse_input(event:Event, polar_coordinates:MutableSequence) -> bool:
	"""Updates object rotation and zoom from mouse interactions.
	Returns whether the screen needs to update.
	"""

	if event.type == pg.MOUSEBUTTONUP:
		pg.event.set_blocked(pg.MOUSEMOTION)
		return False

	if event.type == pg.MOUSEBUTTONDOWN:
		pg.event.set_allowed(pg.MOUSEMOTION)
		# Calling get_rel() once makes the next call relative to current position
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
		polar_coordinates[0] *= (1 + event.y * 0.15)
		return True

	return False

def start_engine(title:str, size:tuple[int,int], color:tuple[int,int,int]) -> SurfaceType:
	pg.init()
	pg.display.set_caption(title)
	window = pg.display.set_mode(size=size)
	window.fill(color)
	# We only need mouse movement when the mouse button is held down.
	pg.event.set_blocked(pg.MOUSEMOTION)
	return window

if __name__=="__main__":
	print()
	main()
	# from util import timing
	# timing.line(main, [draw_polygons])

#Non-standard Libraries:
# pygame

import pygame
from enum import Enum
from collections import namedtuple
import random
from collections.abc import Callable

Pos = namedtuple('Pos', ['x', 'y'])

class BlockType(Enum):
    Empty = 'E'
    Body = 'B'
    Head = 'H'
    Apple = 'A'

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class Action(Enum):
    LEFT = -1
    RIGHT = 1
    NOOP = 0

class BodyType(Enum):
    STRAIGHT_H = 1
    STRAIGHT_V = 2
    CORNER_TL = 3
    CORNER_TR = 4
    CORNER_BL = 5
    CORNER_BR = 6

possible_moves = {
    Direction.NORTH: {
        pygame.K_a: Action.LEFT,
        pygame.K_LEFT: Action.LEFT,
        pygame.K_d: Action.RIGHT,
        pygame.K_RIGHT: Action.RIGHT,
        pygame.K_w: Action.NOOP,
        pygame.K_UP: Action.NOOP
    },
    Direction.SOUTH: {
        pygame.K_a: Action.RIGHT,
        pygame.K_LEFT: Action.RIGHT,
        pygame.K_d: Action.LEFT,
        pygame.K_RIGHT: Action.LEFT,
        pygame.K_s: Action.NOOP,
        pygame.K_DOWN: Action.NOOP
    },
    Direction.EAST: {
        pygame.K_s: Action.RIGHT,
        pygame.K_DOWN: Action.RIGHT,
        pygame.K_d: Action.NOOP,
        pygame.K_RIGHT: Action.NOOP,
        pygame.K_w: Action.LEFT,
        pygame.K_UP: Action.LEFT
    },
    Direction.WEST: {
        pygame.K_a: Action.NOOP,
        pygame.K_LEFT: Action.NOOP,
        pygame.K_s: Action.LEFT,
        pygame.K_DOWN: Action.LEFT,
        pygame.K_w: Action.RIGHT,
        pygame.K_UP: Action.RIGHT
    },
}

class Snake:
    def __init__(self, initial_pos: Pos):
        self.facing = Direction.NORTH
        self.head_pos = initial_pos
        self.body = [Pos(initial_pos.x, y) for y in [initial_pos.y + 1, initial_pos.y + 2]]
        self.body_types = [BodyType.STRAIGHT_V for __ in range(2)]
        self.queued_move = Action.NOOP
        self.apple_eaten = False
    
    @staticmethod
    def copy(old_snake: "Snake"):
        new_snake = Snake(old_snake.head_pos)
        new_snake.body = old_snake.body.copy()
        new_snake.body_types = old_snake.body_types.copy()
        new_snake.apple_eaten = old_snake.apple_eaten
        new_snake.facing = old_snake.facing
        return new_snake
    
    def eat_apple(self):
        self.apple_eaten = True
    
    def predict_move(self, move: Action):
        new_move = Direction((self.facing.value + move.value) % 4)
        delta = {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0)
        }[new_move]

        new_pos = Pos(self.head_pos.x + delta[0], self.head_pos.y + delta[1])
        return new_pos

    def move(self):
        new_move = Direction((self.facing.value + self.queued_move.value) % 4)
        delta = {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0)
        }[new_move]

        new_pos = Pos(self.head_pos.x + delta[0], self.head_pos.y + delta[1])
        
        bodyType_possibilities = {
            (Direction.NORTH, Direction.NORTH): BodyType.STRAIGHT_V,
            (Direction.SOUTH, Direction.SOUTH): BodyType.STRAIGHT_V,
            (Direction.EAST, Direction.EAST): BodyType.STRAIGHT_H,
            (Direction.WEST, Direction.WEST): BodyType.STRAIGHT_H,
            (Direction.EAST, Direction.NORTH): BodyType.CORNER_TL,
            (Direction.WEST, Direction.NORTH): BodyType.CORNER_TR,
            (Direction.EAST, Direction.SOUTH): BodyType.CORNER_BL,
            (Direction.WEST, Direction.SOUTH): BodyType.CORNER_BR,
            (Direction.NORTH, Direction.EAST): BodyType.CORNER_BR,
            (Direction.NORTH, Direction.WEST): BodyType.CORNER_BL,
            (Direction.SOUTH, Direction.EAST): BodyType.CORNER_TR,
            (Direction.SOUTH, Direction.WEST): BodyType.CORNER_TL
        }

        new_bodyType = bodyType_possibilities.get((new_move, self.facing), None)
        if new_bodyType is None:
            raise Exception(f'Unexpected direction pair: {self.facing.name}, {new_move.name}')
        
        self.body.insert(0, self.head_pos)
        self.body_types.insert(0, new_bodyType)
        if self.apple_eaten:
            self.apple_eaten = False
        else:
            self.body.pop()
            self.body_types.pop()
        self.head_pos = new_pos

        self.queued_move = Action.NOOP
        self.facing = new_move

class Board:
    def __init__(self, size: tuple[int]=(10, 10)):
        self.bounds = ((0, size[0]), (0, size[1]))
        self.snake = Snake(Pos(size[0] // 2 - 1, size[1] // 2 - 1))
        self.all_spaces = set(Pos(x, y) for x in range(size[0]) for y in range(size[1]))
        self.apple_pos: Pos | None = None
        self.generate_apple()
        self.is_playing = True

    def generate_apple(self):
        body_set = set(self.snake.body + [self.snake.head_pos])
        possibilities = self.all_spaces ^ body_set
        if possibilities:
            self.apple_pos = random.choice(tuple(possibilities))
        else:
            self.apple_pos = None
    
    def validate_state(self):
        if self.snake.head_pos in self.snake.body:
            self.is_playing = False
        
        if not self.bounds[0][0] <= self.snake.head_pos.x < self.bounds[0][1]:
            self.is_playing = False
        
        if not self.bounds[1][0] <= self.snake.head_pos.y < self.bounds[1][1]:
            self.is_playing = False
        
        if self.snake.head_pos == self.apple_pos:
            self.snake.eat_apple()
            self.generate_apple()
        
        if self.apple_pos is None:
            self.is_playing = False
        
        return self.is_playing
    
    def step(self, move = None):
        if self.is_playing:
            snake_copy = Snake.copy(self.snake)
            if move:
                self.snake.queued_move = move
            self.snake.move()
            if not self.validate_state() and self.apple_pos is not None:
                self.snake = snake_copy

    @staticmethod
    def copy(old_board: "Board"):
        new_board = Board()
        new_board.__dict__ = old_board.__dict__.copy()
        new_board.snake = Snake.copy(old_board.snake)
        return new_board

class Game:
    def __init__(self, size = (10, 10), step_time = 0.5, gridline_size = 1, padding_size = 3, border_size = 7, snake_width = 3):
        self.board = Board(size)
        self.play_size = size
        self.gridline_size = gridline_size
        self.padding_size = padding_size
        self.border_size = border_size
        self.block_size = 30
        self.snake_width = snake_width
        self.width = (size[0] * 2) * self.padding_size + (size[0] - 1) * self.gridline_size + size[0] * self.block_size + 2 * self.border_size
        self.height = (size[1] * 2) * self.padding_size + (size[1] - 1) * self.gridline_size + size[1] * self.block_size + 2 * self.border_size
        # print(self.width, self.height)
        self.surface = pygame.Surface((self.width, self.height))
        self.snake_colour = (104, 255, 104)
        self.snake_border_colour = (0, 147, 0)
        self.step_time = step_time
    
    def step(self):
        self.board.step()
    
    def render(self, clear=True):
        if clear:
            self.surface.fill((0, 0, 0))

        border_mid = (self.border_size + 1) // 2
        # Draw border
        pygame.draw.line(self.surface, (255, 255, 255), (0, border_mid - 1), (self.width, border_mid - 1), self.border_size)
        # pygame.draw.line(self.surface, (255, 0, 0), (0, border_mid - 1), (self.width, border_mid), 1)

        pygame.draw.line(self.surface, (255, 255, 255), (0, self.height - border_mid), (self.width, self.height - border_mid), self.border_size)
        # pygame.draw.line(self.surface, (255, 0, 0), (0, self.height - border_mid), (self.width, self.height - border_mid), 1)

        pygame.draw.line(self.surface, (255, 255, 255), (border_mid - 1, 0), (border_mid - 1, self.height), self.border_size)
        # pygame.draw.line(self.surface, (255, 0, 0), (border_mid - 1, 0), (border_mid, self.height), 1)

        pygame.draw.line(self.surface, (255, 255, 255), (self.width - border_mid, 0), (self.width - border_mid, self.height), self.border_size)
        # pygame.draw.line(self.surface, (255, 0, 0), (self.width - border_mid, 0), (self.width - border_mid, self.height), 1)

        # Draw Gridlines

        # Vertical
        curr_xpos = self.border_size + self.padding_size * 2 + self.block_size
        mid_grid = (self.gridline_size + 1) // 2 - 1
        for __ in range(1, self.play_size[0]):
            mid_pos = curr_xpos + mid_grid
            pygame.draw.line(self.surface, (255, 255, 255), (mid_pos, self.border_size), (mid_pos, self.height - 1 - self.border_size), self.gridline_size)
            curr_xpos += self.gridline_size + self.padding_size * 2 + self.block_size
        
        # Horizontal
        curr_ypos = self.border_size + self.padding_size * 2 + self.block_size
        mid_grid = (self.gridline_size + 1) // 2 - 1
        for __ in range(1, self.play_size[1]):
            mid_pos = curr_ypos + mid_grid
            pygame.draw.line(self.surface, (255, 255, 255), (self.border_size, mid_pos), (self.width - 1 - self.border_size, mid_pos), self.gridline_size)
            curr_ypos += self.gridline_size + self.padding_size * 2 + self.block_size
        
        # Draw Body
        border_types = {
            'TOP': (BodyType.STRAIGHT_H, BodyType.CORNER_TL, BodyType.CORNER_TR, Direction.SOUTH),
            'LEFT': (BodyType.STRAIGHT_V, BodyType.CORNER_TL, BodyType.CORNER_BL, Direction.EAST),
            'RIGHT': (BodyType.STRAIGHT_V, BodyType.CORNER_TR, BodyType.CORNER_BR, Direction.WEST),
            'BOTTOM': (BodyType.STRAIGHT_H, BodyType.CORNER_BL, BodyType.CORNER_BR, Direction.NORTH)
        }
        corners = (
            (0, 0),
            (0, self.block_size - self.snake_width),
            (self.block_size - self.snake_width, self.block_size - self.snake_width),
            (self.block_size - self.snake_width, 0))
        border_corner_inds = {
            'TOP': (0, self.block_size, self.snake_width),
            'LEFT': (0, self.snake_width, self.block_size),
            'RIGHT': (3, self.snake_width, self.block_size),
            'BOTTOM': (1, self.block_size, self.snake_width)
        }
        for position, bodyType in zip(self.board.snake.body, self.board.snake.body_types, strict=True):
            x_pos = self.border_size + position.x * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
            y_pos = self.border_size + position.y * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
            out_rect = pygame.Rect(x_pos, y_pos, self.block_size, self.block_size)
            pygame.draw.rect(self.surface, self.snake_colour, out_rect)

            
            # Default Corners
            for offset in corners:
                new_rect = pygame.Rect(x_pos + offset[0], y_pos + offset[1], self.snake_width, self.snake_width)
                pygame.draw.rect(self.surface, self.snake_border_colour, new_rect)
            
            # Borders
            for border_dir, applicable_types in border_types.items():
                if bodyType in applicable_types:
                    corner_ind, w, h = border_corner_inds[border_dir]
                    corner_val = corners[corner_ind]
                    new_rect = pygame.Rect(x_pos + corner_val[0], y_pos + corner_val[1], w, h)
                    pygame.draw.rect(self.surface, self.snake_border_colour, new_rect)
        
        # Head handling
        x_pos = self.border_size + self.board.snake.head_pos.x * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
        y_pos = self.border_size + self.board.snake.head_pos.y * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
        out_rect = pygame.Rect(x_pos, y_pos, self.block_size, self.block_size)
        pygame.draw.rect(self.surface, self.snake_colour if self.board.is_playing else (255, 0, 0), out_rect)
        for border_dir, applicable_types in border_types.items():
            if self.board.snake.facing not in applicable_types:
                corner_ind, w, h = border_corner_inds[border_dir]
                corner_val = corners[corner_ind]
                new_rect = pygame.Rect(x_pos + corner_val[0], y_pos + corner_val[1], w, h)
                pygame.draw.rect(self.surface, self.snake_border_colour, new_rect)

        # Render apple
        if self.board.apple_pos:
            x_pos = self.border_size + self.board.apple_pos.x * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
            y_pos = self.border_size + self.board.apple_pos.y * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
            out_rect = pygame.Rect(x_pos, y_pos, self.block_size, self.block_size)
            pygame.draw.rect(self.surface, (255,0,0), out_rect)

class AgentGame(Game):
    def __init__(self, action_func: Callable[[Board], list[Action]],size = (10, 10), step_time = 0.5,
                gridline_size = 1, padding_size = 3, border_size = 7, snake_width = 3, pathfinding_storage_count = 0,
                initial_move_func = None, watchdog = None):
        super().__init__(size, step_time, gridline_size, padding_size, border_size, snake_width)
        self.action_func = action_func
        self.action_queue: list[Action] = []
        self.pathfinding_start_poses: list[Pos] = []
        self.max_pathfinding_size = pathfinding_storage_count
        self.what_if_board: Board | None = None
        self.move_sequence_call = watchdog.control if watchdog is not None else lambda x: ...
        self.store_state = watchdog.store if watchdog is not None else lambda x: ...
        if initial_move_func is not None:
            self.action_queue = initial_move_func(self.board)
    
    def step(self, full_render = None):
        if self.board.is_playing:
            if not self.action_queue:
                self.move_sequence_call(True)
                self.action_queue = self.action_func(self.board, renderer = full_render)
                self.move_sequence_call(False)
                self.pathfinding_start_poses.insert(0, self.board.snake.head_pos)
                while len(self.pathfinding_start_poses) > self.max_pathfinding_size:
                    self.pathfinding_start_poses.pop()
            else:
                self.what_if_board = None
            self.board.snake.queued_move = self.action_queue.pop(0)
            self.board.step()
            self.store_state(self.board)
    
    def render(self):
        self.surface.fill((0, 0, 0))

        what_if_colour = (255, 148, 41)
        what_if_border = (138, 79, 21)
        if self.what_if_board is not None:
            border_types = {
                'TOP': (BodyType.STRAIGHT_H, BodyType.CORNER_TL, BodyType.CORNER_TR, Direction.SOUTH),
                'LEFT': (BodyType.STRAIGHT_V, BodyType.CORNER_TL, BodyType.CORNER_BL, Direction.EAST),
                'RIGHT': (BodyType.STRAIGHT_V, BodyType.CORNER_TR, BodyType.CORNER_BR, Direction.WEST),
                'BOTTOM': (BodyType.STRAIGHT_H, BodyType.CORNER_BL, BodyType.CORNER_BR, Direction.NORTH)
            }
            corners = (
                (0, 0),
                (0, self.block_size - self.snake_width),
                (self.block_size - self.snake_width, self.block_size - self.snake_width),
                (self.block_size - self.snake_width, 0))
            border_corner_inds = {
                'TOP': (0, self.block_size, self.snake_width),
                'LEFT': (0, self.snake_width, self.block_size),
                'RIGHT': (3, self.snake_width, self.block_size),
                'BOTTOM': (1, self.block_size, self.snake_width)
            }

            # print(self.what_if_board.snake.body)
            # print(self.board.snake.body)

            for position, bodyType in zip(self.what_if_board.snake.body, self.what_if_board.snake.body_types, strict=True):
                x_pos = self.border_size + position.x * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
                y_pos = self.border_size + position.y * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
                out_rect = pygame.Rect(x_pos, y_pos, self.block_size, self.block_size)
                pygame.draw.rect(self.surface, what_if_colour, out_rect)

                
                # Default Corners
                for offset in corners:
                    new_rect = pygame.Rect(x_pos + offset[0], y_pos + offset[1], self.snake_width, self.snake_width)
                    pygame.draw.rect(self.surface, self.snake_border_colour, new_rect)
                
                # Borders
                for border_dir, applicable_types in border_types.items():
                    if bodyType in applicable_types:
                        corner_ind, w, h = border_corner_inds[border_dir]
                        corner_val = corners[corner_ind]
                        new_rect = pygame.Rect(x_pos + corner_val[0], y_pos + corner_val[1], w, h)
                        pygame.draw.rect(self.surface, what_if_border, new_rect)
            
            # Head handling
            x_pos = self.border_size + self.what_if_board.snake.head_pos.x * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
            y_pos = self.border_size + self.what_if_board.snake.head_pos.y * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
            out_rect = pygame.Rect(x_pos, y_pos, self.block_size, self.block_size)
            pygame.draw.rect(self.surface, what_if_colour if self.board.is_playing else (255, 0, 0), out_rect)
            for border_dir, applicable_types in border_types.items():
                if self.what_if_board.snake.facing not in applicable_types:
                    corner_ind, w, h = border_corner_inds[border_dir]
                    corner_val = corners[corner_ind]
                    new_rect = pygame.Rect(x_pos + corner_val[0], y_pos + corner_val[1], w, h)
                    pygame.draw.rect(self.surface, what_if_border, new_rect)

        super().render(clear = False)

        for ind in range(-1, -len(self.pathfinding_start_poses) - 1, -1):
            position = self.pathfinding_start_poses[ind]
            gradient_index = len(self.pathfinding_start_poses) + ind
            num = 255 - 255 * (gradient_index / self.max_pathfinding_size)
            colour = (num, num, num)
            x_pos = self.border_size + position.x * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
            y_pos = self.border_size + position.y * (self.padding_size * 2 + self.block_size + self.gridline_size) + self.padding_size
            out_rect = pygame.Rect(x_pos, y_pos, self.block_size, self.block_size)
            pygame.draw.rect(self.surface, colour, out_rect)
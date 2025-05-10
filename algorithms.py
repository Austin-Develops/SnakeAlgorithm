from snake import *

# Pre research
def weighted_random_agent_with_check(board: Board, renderer = ...):
    import random
    def alter_weight(weights_dict: dict[Action, float], target_action: Action, offset: float):
        # print(f'Altered {target_action}: {offset}')
        for action in weights_dict:
            if action == target_action:
                weights_dict[action] += offset
            else:
                weights_dict[action] -= offset / 2
    
    def determine_weights(board: Board):
        weights = {
            Action.LEFT: 1/3,
            Action.RIGHT: 1/3,
            Action.NOOP: 1/3
        }
        
        if board.snake.head_pos.y > board.apple_pos.y: # Snake Head is below apple
            if board.snake.facing == Direction.NORTH:
                alter_weight(weights, Action.NOOP, 0.25)
            elif board.snake.facing == Direction.EAST:
                alter_weight(weights, Action.LEFT, 0.3)
            elif board.snake.facing == Direction.WEST:
                alter_weight(weights, Action.RIGHT, 0.3)
            if board.snake.facing == Direction.SOUTH:
                alter_weight(weights, Action.NOOP, -0.3)

        elif board.snake.head_pos.y < board.apple_pos.y: # Snake Head is above apple
            if board.snake.facing == Direction.NORTH:
                alter_weight(weights, Action.NOOP, -0.3)
            elif board.snake.facing == Direction.EAST:
                alter_weight(weights, Action.RIGHT, 0.3)
            elif board.snake.facing == Direction.WEST:
                alter_weight(weights, Action.LEFT, 0.3)
            if board.snake.facing == Direction.SOUTH:
                alter_weight(weights, Action.NOOP, 0.25)
        
        if board.snake.head_pos.x > board.apple_pos.x: # Snake is to the right of apple
            if board.snake.facing == Direction.NORTH:
                alter_weight(weights, Action.LEFT, 0.3)
            elif board.snake.facing == Direction.EAST:
                alter_weight(weights, Action.NOOP, -0.3)
            elif board.snake.facing == Direction.WEST:
                alter_weight(weights, Action.NOOP, 0.25)
            if board.snake.facing == Direction.SOUTH:
                alter_weight(weights, Action.RIGHT, 0.3)

        elif board.snake.head_pos.x < board.apple_pos.x: # Snake is to the left of apple
            if board.snake.facing == Direction.NORTH:
                alter_weight(weights, Action.RIGHT, 0.3)
            elif board.snake.facing == Direction.EAST:
                alter_weight(weights, Action.NOOP, 0.25)
            elif board.snake.facing == Direction.WEST:
                alter_weight(weights, Action.NOOP, -0.3)
            if board.snake.facing == Direction.SOUTH:
                alter_weight(weights, Action.LEFT, 0.3)
        
        weights = {k: v - min(0, min(weights.values())) for k, v in weights.items()} # Ensure no negative weights
        
        return weights
    
    action_list = []
    curr_board = Board.copy(board)
    
    must_return = False
    for __ in range(3):
        possibilities = [Action.NOOP, Action.LEFT, Action.RIGHT]
        weights = determine_weights(curr_board)
        
        while possibilities:
            choice = None
            num = random.random() * sum([weights[p] for p in possibilities])
            curr_sum = 0
            for action in possibilities:
                if num <= curr_sum + weights[action]:
                    choice = action
                    break
                curr_sum += weights[action]
            
            if choice is None:
                # print('Failed to find a valid choice. Sum =', curr_sum)
                return action_list + [Action.NOOP]
            
            # print('Weights:', weights, '\nChoice:', choice)

            board_copy = Board.copy(curr_board)
            apple_pos = board_copy.apple_pos
            board_copy.step(choice)
            if not board_copy.is_playing:
                possibilities.remove(choice)
                board_copy = Board.copy(curr_board)
                # print('Failed')
            else:
                action_list.append(choice)
                if board_copy.snake.head_pos == apple_pos:
                    must_return = True
                curr_board = board_copy
                # print('Success')
                break
        
        if must_return:
            break
        
        # print('-------\n')
        if not possibilities:
            return action_list + [Action.NOOP]
    
    return action_list

# Pre research
def weighted_random_agent_recursive_check(board: Board, *, max_depth = None, depth = None, apple_pos = None, renderer=None):      
    if max_depth is None:
        max_depth = round(len(board.snake.body) * 1.2)
    
    if depth is None:
        depth = max_depth
    
    depth = min(depth, max_depth)

    if depth <= 0:
        return ([], True) if depth != max_depth else []
    
    import random
    def alter_weight(weights_dict: dict[Action, float], target_action: Action, offset: float):
        # print(f'Altered {target_action}: {offset}')
        for action in weights_dict:
            if action == target_action:
                weights_dict[action] += offset
            else:
                weights_dict[action] -= offset / 2
    
    def determine_weights(board: Board):
        weights = {
            Action.LEFT: 1/3,
            Action.RIGHT: 1/3,
            Action.NOOP: 1/3
        }
        
        if board.snake.head_pos.y > board.apple_pos.y: # Snake Head is below apple
            if board.snake.facing == Direction.NORTH:
                alter_weight(weights, Action.NOOP, 0.25)
            elif board.snake.facing == Direction.EAST:
                alter_weight(weights, Action.LEFT, 0.3)
            elif board.snake.facing == Direction.WEST:
                alter_weight(weights, Action.RIGHT, 0.3)
            if board.snake.facing == Direction.SOUTH:
                alter_weight(weights, Action.NOOP, -0.3)

        elif board.snake.head_pos.y < board.apple_pos.y: # Snake Head is above apple
            if board.snake.facing == Direction.NORTH:
                alter_weight(weights, Action.NOOP, -0.3)
            elif board.snake.facing == Direction.EAST:
                alter_weight(weights, Action.RIGHT, 0.3)
            elif board.snake.facing == Direction.WEST:
                alter_weight(weights, Action.LEFT, 0.3)
            if board.snake.facing == Direction.SOUTH:
                alter_weight(weights, Action.NOOP, 0.25)
        
        if board.snake.head_pos.x > board.apple_pos.x: # Snake is to the right of apple
            if board.snake.facing == Direction.NORTH:
                alter_weight(weights, Action.LEFT, 0.3)
            elif board.snake.facing == Direction.EAST:
                alter_weight(weights, Action.NOOP, -0.3)
            elif board.snake.facing == Direction.WEST:
                alter_weight(weights, Action.NOOP, 0.25)
            if board.snake.facing == Direction.SOUTH:
                alter_weight(weights, Action.RIGHT, 0.3)

        elif board.snake.head_pos.x < board.apple_pos.x: # Snake is to the left of apple
            if board.snake.facing == Direction.NORTH:
                alter_weight(weights, Action.RIGHT, 0.3)
            elif board.snake.facing == Direction.EAST:
                alter_weight(weights, Action.NOOP, 0.25)
            elif board.snake.facing == Direction.WEST:
                alter_weight(weights, Action.NOOP, -0.3)
            if board.snake.facing == Direction.SOUTH:
                alter_weight(weights, Action.LEFT, 0.3)
        
        weights = {k: v - min(0, min(weights.values())) for k, v in weights.items()} # Ensure no negative weights
        
        return weights
    
    mhd = lambda posa, posb: abs(posa.x - posb.x) + abs(posa.y - posb.y)
    get_farthest_corner = lambda pos: max([Pos(board.bounds[0][i], board.bounds[1][j]) for i in range(2) for j in range(2)], key=lambda corner: mhd(pos, corner))
    
    curr_board = Board.copy(board)

    if apple_pos is not None:
        curr_board.apple_pos = apple_pos
    
    possibilities = [Action.NOOP, Action.LEFT, Action.RIGHT]
    weights = determine_weights(curr_board)

    # if apple_pos is None:
    #     print('\t', curr_board.apple_pos)
    # else:
    #     print(curr_board.apple_pos)
    
    # print('>', depth, max_depth, apple_pos is None)

    while possibilities:
        if renderer is not None:
            # print('Rendered')
            renderer(board)

        choice = None
        num = random.random() * sum([weights[p] for p in possibilities])
        curr_sum = 0
        for action in possibilities:
            if num <= curr_sum + weights[action]:
                choice = action
                break
            curr_sum += weights[action]
        
        if choice is None:
            # print('Failed to find a valid choice. Sum =', curr_sum)
            return ([Action.NOOP], False) if depth != max_depth else [Action.NOOP]
        
        # print('Weights:', weights, '\nChoice:', choice)

        board_copy = Board.copy(curr_board) 
        board_copy.step(choice)
        if not board_copy.is_playing:
            possibilities.remove(choice)
            board_copy = Board.copy(curr_board)
            # print('Failed')
        else:
            if board_copy.snake.head_pos == board.apple_pos and apple_pos is None:
                # print('Real Apple located')
                path_result = []
                __, outcome_result = weighted_random_agent_recursive_check(board_copy, depth=len(board.snake.body) + 2, max_depth=len(board.snake.body) + 3, apple_pos=get_farthest_corner(board.snake.head_pos),
                                                                           renderer=renderer)
            elif board_copy.snake.head_pos == board.apple_pos:
                path_result, outcome_result = weighted_random_agent_recursive_check(board_copy, depth=depth - 1, max_depth=max_depth, apple_pos=get_farthest_corner(board.snake.head_pos),
                                                                                    renderer=renderer)
            else:
                path_result, outcome_result = weighted_random_agent_recursive_check(board_copy, depth=depth - 1, max_depth=max_depth, apple_pos=apple_pos,
                                                                                    renderer=renderer)
            
            # print(depth, board.snake.head_pos, path_result, outcome_result)

            if outcome_result:
                # print('Success')
                return ([choice] + path_result, True) if depth != max_depth else [choice] + path_result
            # print('Inside failure')
            elif (depth + 1) % 2 != 0:
                return ([Action.NOOP], False) if depth != max_depth else [Action.NOOP]
            possibilities.remove(choice)
    
    # print('-------\n')
    if not possibilities:
        return ([Action.NOOP], False) if depth != max_depth else [Action.NOOP]
    
    return ":P" # You shouldn't be here


def hamiltonian_cycle_initializer(board: Board):
    board_size = board.bounds[0][1], board.bounds[1][1]
    actions = [Action.RIGHT] + [Action.NOOP] * (board_size[0] - board.snake.head_pos.x - 2)  + [Action.LEFT]
    actions += [Action.NOOP] * (board.snake.head_pos.y - 1)
    return actions


def hamiltonian_cycle(board: Board, renderer = ...):
    # This should be called when the thing is in the corner
    board_size = board.bounds[0][1], board.bounds[1][1]
    if board_size[1] % 2 == 0:
        return hamiltonian_cycle_even(board) 
    else:
        return [Action.NOOP] * 20

def hamiltonian_cycle_even(board: Board):
    # Snake is in the top right corner facing north
    board_size = board.bounds[0][1], board.bounds[1][1]
    actions = [Action.LEFT] + [Action.NOOP] * (board_size[0] - 2) + [Action.LEFT, Action.NOOP]

    hall_actions = [Action.NOOP] * (board_size[1] - 3)
    left_trans = [Action.LEFT] * 2
    right_trans = [Action.RIGHT] * 2

    for __ in range(board_size[0] // 2 - 1):
        actions += hall_actions + left_trans + hall_actions + right_trans
    
    actions += hall_actions + left_trans + hall_actions + [Action.NOOP]
    return actions

class Evolved_Hamiltonian_Cycle_Solver:
    def __init__(self):
        self.excluded_spaces = set()
        self.max_size = 0

    @property
    def size_left(self):
        return self.max_size - len(self.excluded_spaces)
    
    def create_fake_board(self, board: Board, board_size):
        fake_board = Board(board_size)
        fake_board.snake.head_pos = Pos(board_size[0] - 1, 0)
        fake_board.snake.body = [Pos(board_size[0] - 1, 1), Pos(board_size[0] - 1, 2)]
        fake_board.apple_pos = board.apple_pos
        return fake_board

    def correct_next_action(self, fake_direction: Direction, true_direction: Direction, fake_action: Action):
        # print(f'Correcting {fake_action} ({fake_direction} to {true_direction})')
        if true_direction == fake_direction:
            return fake_action
        
        if (true_direction.value - fake_direction.value) % 2 == 0:
            return Action(-fake_action.value)
        
        if (fake_direction == Direction.EAST and true_direction == Direction.SOUTH):
            return Action.NOOP
        
        if (fake_direction == Direction.NORTH and true_direction == Direction.EAST):
            if fake_action == Action.RIGHT:
                return Action.NOOP
            elif fake_action == Action.NOOP:
                return Action.LEFT
        
        if (fake_direction == Direction.SOUTH and true_direction == Direction.EAST):
            if fake_action == Action.LEFT:
                return Action.NOOP
            elif fake_action == Action.NOOP:
                return Action.RIGHT
        
        raise Exception(f"Forgot to include some case: {fake_direction}, {true_direction}, {fake_action}")

    def get_topside_excluded(self, board_size, x, y):
        temp_excluded = set(Pos(a, b) for a in range(x) for b in range(board_size[1]))
        if x % 2 == 1:
            temp_excluded = temp_excluded | set(Pos(x, b) for b in range(2, board_size[1]))
        
        # print(len(self.excluded_spaces | temp_excluded))

        return self.max_size - len(self.excluded_spaces | temp_excluded)

    def get_upper_boundary_excluded(self, board_size, x, y):
        temp_excluded = set(Pos(a, b) for a in (x, x+1) for b in range(1, y))
        return self.max_size - len(self.excluded_spaces | temp_excluded)
    
    def get_lower_boundary_excluded(self, board_size, x, y):
        temp_excluded = set(Pos(a, b) for a in (x, x+1) for b in range(y + 1, board_size[1]))
        return self.max_size - len(self.excluded_spaces | temp_excluded)
    
    def get_east_skip_excluded(self, board_size, x, y):
        # Since it is based off of fake_board (the true cycle), only occurs at y = 1 and y = board_size[1] - 1
        if y == 1:
            temp_excluded = set(Pos(a, b) for a in (x, x+1) for b in range(2, board_size[1]))
        elif y == board_size[1] - 1:
            temp_excluded = set(Pos(a, b) for a in (x, x+1) for b in range(1, y - 1))
        else:
            # print('This case should not be possible')
            return self.size_left
        
        return self.max_size - len(self.excluded_spaces | temp_excluded)

    def hamiltonian_cycle_evolved(self, board: Board, renderer = ...):
        '''This paths to either the apple, or the top right corner'''
        board_size = board.bounds[0][1], board.bounds[1][1]
        self.max_size = board_size[0] * board_size[1]

        fake_board = self.create_fake_board(board, board_size)
        path = hamiltonian_cycle_even(board)
        # print(path)
        path_to_apple = True
        while fake_board.snake.head_pos != board.snake.head_pos:
            if fake_board.snake.head_pos == board.apple_pos:
                path_to_apple = False
            fake_board.step(path.pop(0))
        
        # print(path)
        
        if path_to_apple:
            positions = []
            apple_pather = Board.copy(fake_board)
            cutoff_ind = 0
            while apple_pather.snake.head_pos != board.apple_pos:
                apple_pather.step(path[cutoff_ind])
                positions.append(apple_pather.snake.head_pos)
                cutoff_ind += 1
            path = path[:cutoff_ind]
        
        # print(path)
        
        # print(path)
        
        parser_board = Board.copy(board)
        
        true_path = []

        while path:
            # print(true_path)
            # print(parser_board.snake.head_pos)
            # print(len(self.excluded_spaces))
            fake_move = path.pop(0)
            default_move = self.correct_next_action(fake_board.snake.facing, parser_board.snake.facing, fake_move)
            # print(fake_move, default_move, true_path, fake_board.is_playing, fake_board.snake.head_pos, fake_board.snake.facing, parser_board.snake.facing)
            # print(self.excluded_spaces)

            if fake_board.snake.head_pos.y == 0 and fake_board.snake.head_pos.x not in (0, board_size[0] - 1):
                # print('topside')
                x = fake_board.snake.head_pos.x
                potential_loss = (2 * (board_size[1] - 2) * ((x + 1) // 2)) + (2 * x)
                # print(f'{potential_loss = }')
                if len(parser_board.snake.body) + 1 < self.get_topside_excluded(board_size, *parser_board.snake.head_pos) - int(fake_board.snake.predict_move(Action.LEFT) in self.excluded_spaces):
                    # print('in if')
                    path_old = path.copy()
                    fake_board_old = Board.copy(fake_board)
                    
                    more_excluded = set()

                    if len(path) < potential_loss:
                        # print('Path too short', path)
                        pass

                    else:
                        fake_board.step(fake_move)
                        more_excluded.add(fake_board.snake.head_pos)
                        if fake_board.snake.head_pos == board.apple_pos:
                            fake_board = fake_board_old
                            path = path_old
                            # print('apple found failed')

                        else:
                            skipped_apple = False

                            for ind in range(potential_loss - 1):
                                fake_board.step(path.pop(0))
                                more_excluded.add(fake_board.snake.head_pos)
                                if fake_board.snake.head_pos == board.apple_pos:
                                    fake_board = fake_board_old
                                    path = path_old
                                    skipped_apple = True
                                    # print('Apple found. Failed')
                            
                            fake_board.step(path.pop(0))

                            if not skipped_apple:
                                # print('mid added')
                                true_path.append(Action.LEFT)
                                parser_board.step(Action.LEFT)
                                self.excluded_spaces = self.excluded_spaces | more_excluded
                                self.excluded_spaces.discard(parser_board.snake.head_pos)
                                # print(len(path))
                            
                                continue
            
            if fake_board.snake.facing == Direction.NORTH and fake_board.snake.head_pos.x != board_size[0] - 1 and \
                fake_board.snake.head_pos.y != 1:
                # print('Going up')
                x, y = fake_board.snake.head_pos
                potential_loss = 2 * (y - 1)
                if len(parser_board.snake.body) + 1 < self.get_upper_boundary_excluded(board_size, *parser_board.snake.head_pos) - int(fake_board.snake.predict_move(Action.RIGHT) in self.excluded_spaces):
                    path_old = path.copy()
                    fake_board_old = Board.copy(fake_board)
                    more_excluded = set()

                    if len(path) < potential_loss:
                        pass
                    
                    else:
                        fake_board.step(fake_move)
                        more_excluded.add(fake_board.snake.head_pos)
                        if fake_board.snake.head_pos == board.apple_pos:
                            fake_board = fake_board_old
                            path = path_old
                            # print('Apple found.')
                        
                        else:
                            skipped_apple = False

                            for ind in range(potential_loss - 1):
                                fake_board.step(path.pop(0))
                                more_excluded.add(fake_board.snake.head_pos)
                                if fake_board.snake.head_pos == board.apple_pos:
                                    fake_board = fake_board_old
                                    path = path_old
                                    skipped_apple = True
                                    # print('Apple found.')
                            
                            fake_board.step(path.pop(0))

                            if not skipped_apple:
                                # print('Right skip added')
                                corrected_move = self.correct_next_action(fake_board_old.snake.facing, parser_board.snake.facing, Action.RIGHT)
                                true_path.append(corrected_move)
                                parser_board.step(corrected_move)
                                self.excluded_spaces = self.excluded_spaces | more_excluded
                                self.excluded_spaces.discard(parser_board.snake.head_pos)
                                # print(len(path))
                        
                                continue

            if fake_board.snake.facing == Direction.SOUTH and fake_board.snake.head_pos.y != board_size[1] - 1:
                # print('Going down')
                x, y = fake_board.snake.head_pos
                potential_loss = 2 * (board_size[1] - 1 - y)
                if len(parser_board.snake.body) + 1 < self.get_lower_boundary_excluded(board_size, *parser_board.snake.head_pos) - int(fake_board.snake.predict_move(Action.LEFT) in self.excluded_spaces):
                    path_old = path.copy()
                    fake_board_old = Board.copy(fake_board)
                    more_excluded = set()

                    if len(path) < potential_loss:
                        pass
                    
                    else:
                        fake_board.step(fake_move)
                        more_excluded.add(fake_board.snake.head_pos)
                        if fake_board.snake.head_pos == board.apple_pos:
                            fake_board = fake_board_old
                            path = path_old
                            # print('Apple found.')
                        
                        else:
                            skipped_apple = False

                            for ind in range(potential_loss - 1):
                                fake_board.step(path.pop(0))
                                more_excluded.add(fake_board.snake.head_pos)
                                if fake_board.snake.head_pos == board.apple_pos:
                                    fake_board = fake_board_old
                                    path = path_old
                                    skipped_apple = True
                                    # print('Apple found')
                            
                            fake_board.step(path.pop(0))

                            if not skipped_apple:
                                # print('Left skip added.')
                                corrected_move = self.correct_next_action(fake_board_old.snake.facing, parser_board.snake.facing, Action.LEFT)
                                true_path.append(corrected_move)
                                parser_board.step(corrected_move)
                                self.excluded_spaces = self.excluded_spaces | more_excluded
                                self.excluded_spaces.discard(parser_board.snake.head_pos)
                                # print(len(path))
                        
                                continue
            
            if fake_board.snake.facing == Direction.EAST:
                # print('Going right')
                # print(parser_board.snake.head_pos)
                potential_loss = 2 * (board_size[1] - 2)
                # print(len(parser_board.snake.body), self.get_east_skip_excluded(board_size, *parser_board.snake.head_pos)) 
                # print(Pos(3, 1) in self.excluded_spaces)
                if len(parser_board.snake.body) + 1 < self.get_east_skip_excluded(board_size, *parser_board.snake.head_pos) - int(fake_board.snake.predict_move(Action.NOOP) in self.excluded_spaces):
                    path_old = path.copy()
                    fake_board_old = Board.copy(fake_board)
                    more_excluded = set()
                    
                    if len(path) < potential_loss:
                        pass
                    
                    else:
                        fake_board.step(fake_move)
                        more_excluded.add(fake_board.snake.head_pos)
                        if fake_board.snake.head_pos == board.apple_pos:
                            fake_board = fake_board_old
                            path = path_old
                            # print('Apple found')
                        
                        else:
                            skipped_apple = False
                            
                            for ind in range(potential_loss):
                                fake_board.step(path.pop(0))
                                more_excluded.add(fake_board.snake.head_pos)
                                if fake_board.snake.head_pos == board.apple_pos:
                                    fake_board = fake_board_old
                                    path = path_old
                                    skipped_apple = True
                                    # print('Apple found.')

                            if not skipped_apple:
                                # print('Noop added')
                                corrected_move = self.correct_next_action(fake_board_old.snake.facing, parser_board.snake.facing, Action.NOOP)
                                true_path.append(corrected_move)
                                parser_board.step(corrected_move)
                                self.excluded_spaces = self.excluded_spaces | more_excluded
                                self.excluded_spaces.discard(parser_board.snake.head_pos)
                                # print(len(path))
                        
                                continue
            
            fake_board.step(fake_move)
            true_path.append(default_move)
            parser_board.step(default_move)
            self.excluded_spaces.discard(parser_board.snake.head_pos)
        
        # print(self.size_left, len(board.snake.body) + 1)
        # print('T:', true_path)
        return true_path


# Pre research
def outside_direct_agent(board: Board, renderer = ...):
    board = Board.copy(board)
    act = lambda a: (board.step(a), a)[1]
    actions = []
    apple_pos = board.apple_pos
    if board.snake.facing in (Direction.EAST, Direction.WEST):
        actions.append(Action.RIGHT)
        board.step(Action.RIGHT)
    if board.snake.facing == Direction.NORTH:
        for __ in range(board.snake.head_pos.y):
            actions += [act(Action.NOOP)]
            if board.snake.head_pos == apple_pos:
                return actions
        correction = Action.LEFT if board.apple_pos.x <= board.snake.head_pos.x else Action.RIGHT
        actions.append(act(Action.LEFT) if board.apple_pos.x <= board.snake.head_pos.x else act(Action.RIGHT))

    else:
        for __ in range(board.bounds[1][1] - board.snake.head_pos.y - 1):
            actions += [act(Action.NOOP)]
            if board.snake.head_pos == apple_pos:
                return actions
        correction = Action.LEFT if board.apple_pos.x > board.snake.head_pos.x else Action.RIGHT
        actions.append(act(Action.LEFT) if board.apple_pos.x > board.snake.head_pos.x else Action.RIGHT)
    
    actions += [act(Action.NOOP) for __ in range(max(board.apple_pos.x - board.snake.head_pos.x, board.snake.head_pos.x - 1 - board.apple_pos.x))]
    actions.append(correction)
    
    # print(actions)

    return actions
from snake import *
from algorithms import *
import sys

def test():
    pygame.init()
    game = Game()
    display = pygame.display.set_mode(game.surface.get_size())
    game.render()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        display.fill((255, 255, 255))
        display.blit(game.surface, (0, 0))
        pygame.display.update()
    
def main():
    pygame.init()
    clock = pygame.time.Clock()
    game = Game((20, 20), step_time=0.1)
    step_timer = game.step_time
    display = pygame.display.set_mode(game.surface.get_size())
    move_queue = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                move_queue.append(event.key)
                if event.key == pygame.K_RETURN and not game.board.is_playing:
                    move_queue = []
                    game = Game((20, 20), step_time=0.1)
                    
        if game.board.is_playing:
            if step_timer <= 0:
                while move_queue:
                    new_action = possible_moves[game.board.snake.facing].get(move_queue.pop(0), None)
                    if new_action is not None:
                        game.board.snake.queued_move = new_action
                        break
                game.step()
                step_timer = game.step_time
            game.render()
        
            display.fill((255, 255, 255))
            display.blit(game.surface, (0, 0))
            pygame.display.update()

            step_timer -= clock.tick(60) / 1000

def main_agent(agent_func, init_func = None,*, run_mode: str = 'N',
               return_results: bool = False, should_render = True,
               watchdog = None):
    # random.seed(3482934) # Comment when doing tests for all
    step_type = run_mode
    enter_step = step_type.lower() == 'y'
    enter_hold_step = step_type.lower() == 'z'
    pygame.init()
    clock = pygame.time.Clock()
    game = AgentGame(agent_func, (20, 20), step_time=0.01, snake_width=7, pathfinding_storage_count=0, initial_move_func=init_func,
                     watchdog=watchdog)#, gridline_size=0, padding_size=0)
    step_timer = game.step_time
    display = pygame.display.set_mode(game.surface.get_size())

    def render(what_if_board = None):
        if not should_render:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
        if what_if_board is not None:
            game.what_if_board = what_if_board
        game.render()
        display.fill((255, 255, 255))
        display.blit(game.surface, (0, 0))
        pygame.display.update()
        pygame.time.wait(int(game.step_time * 1000))

    render()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            
            if event.type == pygame.KEYDOWN and enter_step:
                if game.board.is_playing:
                    game.step(full_render = render)
                    render()
                    
            
            if event.type == pygame.KEYDOWN and not enter_step:
                if not game.board.is_playing and event.key == pygame.K_1:
                    game = AgentGame(agent_func, (20, 20), step_time=0.00, snake_width=7, pathfinding_storage_count=5, initial_move_func=init_func,
                                     watchdog=watchdog)

        if not enter_step and (not enter_hold_step or pygame.key.get_pressed()[pygame.K_RETURN]):        
            if game.board.is_playing:
                if step_timer <= 0:
                    # game.step(full_render = render)
                    game.step()
                    step_timer = game.step_time
                
                render()
                step_timer -= clock.tick(60) / 1000
            
            elif return_results:
                apples_eaten = len(game.board.snake.body) - 3
                return apples_eaten


if __name__ == '__main__':
    # main()
    # step_type = input('Step on keypress (Y/N): ')
    step_type = 'Z'
    # main_agent(outside_direct_agent, run_mode=step_type)
    # main_agent(weighted_random_agent_with_check, run_mode=step_type)
    # main_agent(weighted_random_agent_recursive_check, run_mode=step_type)
    # main_agent(hamiltonian_cycle, hamiltonian_cycle_initializer, run_mode=step_type)
    main_agent(Evolved_Hamiltonian_Cycle_Solver().hamiltonian_cycle_evolved, hamiltonian_cycle_initializer, run_mode=step_type)
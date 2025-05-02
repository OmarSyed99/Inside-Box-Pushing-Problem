import gym
from gym import spaces
import numpy as np
import pygame

class InsideBoxPushingEnv(gym.Env):
    """
    An environment where a group of agents are inside a box.
    They work together to push the box (a blue circle) to a target.
    One designated agent (agent 0) is faulty: it never contributes push force
    and adds extra mass, slowing movement. If enough agents vote to eliminate it,
    it is removed from the box. Optionally there's a static obstacle agents must avoid.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 grid_size=10,
                 num_agents=4,
                 max_steps=100,
                 base_mass=1.0,
                 faulty_extra_mass=1.0,
                 push_strength=1.0,
                 elimination_threshold=None,
                 target_threshold=0.5,
                 enable_voting=True,
                 enable_obstacle=False):
        super(InsideBoxPushingEnv, self).__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0

        # Physics parameters.
        self.base_mass = base_mass
        self.faulty_extra_mass = faulty_extra_mass
        self.push_strength = push_strength
        self.target_threshold = target_threshold

        # Voting and obstacle toggles
        self.enable_voting = enable_voting
        self.enable_obstacle = enable_obstacle

        # Elimination threshold (default: more than half of agents).
        self.elimination_threshold = (
            elimination_threshold if elimination_threshold is not None
            else self.num_agents // 2
        )

        # Define action space: 0=no-op, 1=up,2=down,3=left,4=right,5=vote
        self.action_space = spaces.MultiDiscrete([6] * num_agents)

        # Observation: [box_x, box_y, target_x, target_y, agent0_alive...agentN_alive]
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(num_agents + 4,),
            dtype=np.float32
        )

        # Rendering parameters.
        self.window_size = 500
        self.cell_size = self.window_size / self.grid_size
        self.screen = None
        self.clock = None

        self.reset()

    def reset(self):
        self.current_step = 0

        # Box starts at a random continuous position
        self.box_pos = np.random.uniform(0, self.grid_size, size=2).astype(np.float32)

        # Target fixed at bottom-right
        self.target_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float32)

        # All agents alive initially
        self.agents_alive = np.ones(self.num_agents, dtype=np.float32)
        self.faulty_agent = 0

        # Obstacle location (integer cell) if enabled
        if self.enable_obstacle:
            while True:
                candidate = np.random.randint(0, self.grid_size, size=2)
                # avoid target cell
                if not np.array_equal(candidate, self.target_pos.astype(int)):
                    self.obstacle_pos = candidate
                    break
        else:
            self.obstacle_pos = None

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.box_pos,
            self.target_pos,
            self.agents_alive
        ]).astype(np.float32)

    def step(self, actions):
        self.current_step += 1
        reward = 0
        info = {}

        # Process votes
        if self.enable_voting:
            votes = sum(1 for i, a in enumerate(actions)
                        if a == 5 and self.agents_alive[i] == 1)
            if votes > self.elimination_threshold:
                self.agents_alive[self.faulty_agent] = 0

        # Compute net push force
        net_force = np.zeros(2, dtype=np.float32)
        for i, action in enumerate(actions):
            if self.agents_alive[i] == 0:
                continue

            force = np.zeros(2, dtype=np.float32)
            if action == 1:
                force[1] -= self.push_strength
            elif action == 2:
                force[1] += self.push_strength
            elif action == 3:
                force[0] -= self.push_strength
            elif action == 4:
                force[0] += self.push_strength

            # Faulty agent never pushes
            if i == self.faulty_agent:
                force[:] = 0

            net_force += force

        # Update box position with simple physics
        effective_mass = self.base_mass + (
            self.faulty_extra_mass if self.agents_alive[self.faulty_agent] == 1 else 0
        )
        displacement = net_force / effective_mass
        new_box_pos = self.box_pos + displacement

        # Prevent box from entering obstacle cell
        if self.enable_obstacle:
            rounded = np.round(new_box_pos).astype(int)
            if not np.array_equal(rounded, self.obstacle_pos):
                self.box_pos = np.clip(new_box_pos, 0, self.grid_size - 1)
        else:
            self.box_pos = np.clip(new_box_pos, 0, self.grid_size - 1)

        # Agents also cannot step into obstacle (handled if you later add agent movement)
        # (Currently agents are inside box and do not move independently.)

        # Check for success
        done = False
        if np.linalg.norm(self.box_pos - self.target_pos) < self.target_threshold:
            reward = 100.0
            done = True
        elif self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (int(self.window_size), int(self.window_size)))
            pygame.display.set_caption("Inside-Box Pushing Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Draw grid
        for x in range(0, int(self.window_size), int(self.cell_size)):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (x, 0), (x, int(self.window_size)))
        for y in range(0, int(self.window_size), int(self.cell_size)):
            pygame.draw.line(self.screen, (200, 200, 200),
                             (0, y), (int(self.window_size), y))

        # Draw obstacle
        if self.enable_obstacle and self.obstacle_pos is not None:
            obs_rect = pygame.Rect(
                self.obstacle_pos[0] * self.cell_size,
                self.obstacle_pos[1] * self.cell_size,
                self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (100, 100, 100), obs_rect)

        # Draw target
        tgt_rect = pygame.Rect(
            self.target_pos[0] * self.cell_size,
            self.target_pos[1] * self.cell_size,
            self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), tgt_rect)

        # Draw box
        box_pixel = (
            int(self.box_pos[0] * self.cell_size),
            int(self.box_pos[1] * self.cell_size)
        )
        box_radius = int(self.cell_size * 1.5)
        pygame.draw.circle(self.screen, (0, 0, 255),
                           box_pixel, box_radius)

        # Draw agents inside box
        alive = [i for i, v in enumerate(self.agents_alive) if v == 1]
        n = len(alive)
        if n > 0:
            inner_r = int(box_radius * 0.5)
            for idx, i in enumerate(alive):
                angle = 2 * np.pi * idx / n
                ax = box_pixel[0] + int(inner_r * np.cos(angle))
                ay = box_pixel[1] + int(inner_r * np.sin(angle))
                color = (255, 0, 0) if i == self.faulty_agent else (0, 0, 0)
                pygame.draw.circle(self.screen, color,
                                   (ax, ay), int(self.cell_size * 0.3))

        # Status text
        font = pygame.font.SysFont("Arial", 16)
        txt = f"Alive: {int(np.sum(self.agents_alive))}"
        surf = font.render(txt, True, (0, 0, 0))
        self.screen.blit(surf, (10, 10))

        if self.enable_voting and self.agents_alive[self.faulty_agent] == 0:
            surf2 = font.render("Faulty removed", True, (255, 0, 0))
            self.screen.blit(surf2, (10, 30))

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
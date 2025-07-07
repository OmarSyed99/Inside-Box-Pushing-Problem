import gym
from gym import spaces
import numpy as np
import pygame


class InsideBoxPushingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 4,
        max_steps: int = 200,
        base_mass: float = 1.0,
        faulty_extra_mass: float = 1.0,
        push_strength: float = 1.0,
        elimination_threshold: int | None = None,
        target_threshold: float = 0.5,
        enable_voting: bool = True,
        enable_obstacle: bool = False,
        random_faulty: bool = True
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.base_mass = base_mass
        self.faulty_extra_mass = faulty_extra_mass
        self.push_strength = push_strength
        self.target_threshold = target_threshold
        self.enable_voting = enable_voting
        self.enable_obstacle = enable_obstacle
        self.random_faulty = random_faulty
        self.elimination_threshold = (
            elimination_threshold if elimination_threshold is not None else num_agents // 2
        )

        self.action_space = spaces.MultiDiscrete([6] * num_agents)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(num_agents + 4,), dtype=np.float32
        )

        self.window_size = 500
        self.cell_size = self.window_size / self.grid_size
        self.screen = None
        self.clock = None

        self.reset()

    def _get_obs(self):
        return np.concatenate([self.box_pos, self.target_pos, self.agents_alive]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.box_pos = np.random.uniform(0, self.grid_size, size=2).astype(np.float32)
        self.target_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float32)
        self.agents_alive = np.ones(self.num_agents, dtype=np.float32)
        if self.random_faulty:
            self.faulty_agent = np.random.randint(self.num_agents)
        else:
            self.faulty_agent = 0
        self.faulty_eliminated_this_episode = False

        if self.enable_obstacle:
            while True:
                cand = np.random.randint(0, self.grid_size, size=2)
                if not np.array_equal(cand, self.target_pos.astype(int)):
                    self.obstacle_pos = cand
                    break
        else:
            self.obstacle_pos = None

        return self._get_obs()

    def step(self, actions):
        self.current_step += 1
        reward = -0.1
        info = {}

        if self.enable_voting:
            votes = sum(
                1 for i, a in enumerate(actions)
                if a == 5 and self.agents_alive[i] == 1
            )
            if (
                votes >= self.elimination_threshold
                and self.agents_alive[self.faulty_agent] == 1
            ):
                self.agents_alive[self.faulty_agent] = 0
                self.faulty_eliminated_this_episode = True
                reward += 5.0

        net_force = np.zeros(2, dtype=np.float32)
        for i, a in enumerate(actions):
            if self.agents_alive[i] == 0:
                continue
            force = np.zeros(2, dtype=np.float32)
            if a == 1:
                force[1] -= self.push_strength
            elif a == 2:
                force[1] += self.push_strength
            elif a == 3:
                force[0] -= self.push_strength
            elif a == 4:
                force[0] += self.push_strength
            if i == self.faulty_agent:
                force[:] = 0
            net_force += force
        net_force = net_force *0.25
        prev_dist = np.linalg.norm(self.box_pos - self.target_pos)
        eff_mass = self.base_mass + (
            self.faulty_extra_mass
            if self.agents_alive[self.faulty_agent] == 1
            else 0
        )
        new_pos = self.box_pos + net_force / eff_mass
        # print(new_pos)

        if self.enable_obstacle and self.obstacle_pos is not None:
            if not np.array_equal(np.round(new_pos).astype(int), self.obstacle_pos):
                self.box_pos = np.clip(new_pos, 0, self.grid_size - 1)
        else:
            self.box_pos = np.clip(new_pos, 0, self.grid_size - 1)

        new_dist = np.linalg.norm(self.box_pos - self.target_pos)
        reward += (prev_dist - new_dist) * 10.0

        done = False
        if new_dist < self.target_threshold:
            reward += 10.0
            done = True
            info["success"] = True
        elif self.current_step >= self.max_steps:
            done = True
            info["success"] = False

        if done and self.agents_alive[self.faulty_agent] == 1:
            reward -= 2.5

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Inside‑Box Pushing")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        for x in range(0, int(self.window_size), int(self.cell_size)):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, int(self.window_size), int(self.cell_size)):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.window_size, y))

        if self.enable_obstacle and self.obstacle_pos is not None:
            obs = pygame.Rect(
                int(self.obstacle_pos[0] * self.cell_size),
                int(self.obstacle_pos[1] * self.cell_size),
                int(self.cell_size),
                int(self.cell_size),
            )
            pygame.draw.rect(self.screen, (100, 100, 100), obs)

        tgt = pygame.Rect(
            int(self.target_pos[0] * self.cell_size),
            int(self.target_pos[1] * self.cell_size),
            int(self.cell_size),
            int(self.cell_size),
        )
        pygame.draw.rect(self.screen, (0, 255, 0), tgt)

        bx, by = (self.box_pos * self.cell_size).astype(int)
        radius = int(self.cell_size * 1.5)
        pygame.draw.circle(self.screen, (0, 0, 255), (bx, by), radius)

        alive_ids = [i for i, a in enumerate(self.agents_alive) if a == 1]
        if alive_ids:
            inner_r = int(radius * 0.5)
            for idx, i in enumerate(alive_ids):
                angle = 2 * np.pi * idx / len(alive_ids)
                ax = bx + int(inner_r * np.cos(angle))
                ay = by + int(inner_r * np.sin(angle))
                color = (255, 0, 0) if i == self.faulty_agent else (0, 0, 0)
                pygame.draw.circle(self.screen, color, (ax, ay), int(self.cell_size * 0.3))

        font = pygame.font.SysFont("Arial", 16)
        self.screen.blit(
            font.render(f"Alive: {int(np.sum(self.agents_alive))}", True, (0, 0, 0)), (10, 10)
        )
        if self.agents_alive[self.faulty_agent] == 0:
            self.screen.blit(font.render("Faulty removed", True, (255, 0, 0)), (10, 30))

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

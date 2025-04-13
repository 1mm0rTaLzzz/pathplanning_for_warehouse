import pygame
import numpy as np
import math
import heapq
from enum import Enum
import random
import time

# Константы
WIDTH, HEIGHT = 1000, 600  # Увеличиваем ширину окна
INFO_WIDTH = 300  # Ширина окна информации
CELL_SIZE = 20
GRID_WIDTH = (WIDTH - INFO_WIDTH) // CELL_SIZE  # Теперь будет больше клеток
GRID_HEIGHT = HEIGHT // CELL_SIZE

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)       # Динамические препятствия
GREEN = (0, 255, 0)     # Точка загрузки
BLUE = (0, 0, 255)      # Робот без груза
YELLOW = (255, 255, 0)  # Стартовая позиция
PURPLE = (128, 0, 128)  # Робот с грузом
GRAY = (128, 128, 128)  # Статические препятствия (стеллажи)
ORANGE = (255, 165, 0)  # Путь
LIGHT_BLUE = (173, 216, 230)  # Фон для проходов
DARK_GREEN = (0, 100, 0)  # Точка выгрузки
BROWN = (139, 69, 19)   # Движущиеся препятствия

# Типы ячеек
class CellType(Enum):
    """Перечисление типов ячеек на карте"""
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PICKUP = 4
    PATH = 5
    ROBOT = 6
    MOVING_OBSTACLE = 7

# Класс движущегося препятствия
class MovingObstacle:
    """Класс для представления движущегося препятствия"""
    
    def __init__(self, pos, grid, speed=0.2):
        """
        Инициализация движущегося препятствия
        
        Args:
            pos: Начальная позиция препятствия
            grid: Сетка карты
            speed: Скорость движения препятствия
        """
        self.pos = pos
        self.prev_pos = pos
        self.grid = grid
        self.direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        self.steps_in_direction = random.randint(3, 8)
        self.current_steps = 0
        self.speed = speed
        self.last_move_time = 0
        self.move_delay = random.uniform(0.3, 0.6)
    
    def update(self, grid, current_time, robot_pos, pickup_pos, dropoff_pos, shelf_positions):
        """
        Обновление состояния препятствия
        
        Args:
            grid: Сетка карты
            current_time: Текущее время
            robot_pos: Позиция робота
            pickup_pos: Позиция точки забора
            dropoff_pos: Позиция точки выгрузки
            shelf_positions: Список позиций стеллажей
            
        Returns:
            bool: True если препятствие переместилось, иначе False
        """
        if current_time - self.last_move_time < self.move_delay:
            return False
        
        if self.current_steps >= self.steps_in_direction:
            possible_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            possible_directions.remove(self.direction)
            self.direction = random.choice(possible_directions)
            self.steps_in_direction = random.randint(3, 8)
            self.current_steps = 0
        
        new_r, new_c = self.pos[0] + self.direction[0], self.pos[1] + self.direction[1]
        
        if (2 <= new_r < GRID_HEIGHT - 2 and 2 <= new_c < GRID_WIDTH - 2 and 
            grid[new_r, new_c] == CellType.EMPTY.value and 
            (new_r, new_c) != robot_pos and 
            (new_r, new_c) != pickup_pos and 
            (new_r, new_c) != dropoff_pos and
            (new_r, new_c) not in shelf_positions):
            
            self.prev_pos = self.pos
            self.pos = (new_r, new_c)
            self.current_steps += 1
            self.last_move_time = current_time
            return True
        else:
            self.direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            self.current_steps = 0
            return False

# Класс для алгоритма A* с возможностью перепланирования
class Pathfinder:
    """Класс для поиска пути с использованием алгоритма A*"""
    
    def __init__(self, grid):
        """
        Инициализация поисковика пути
        
        Args:
            grid: Сетка карты
        """
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
    
    def heuristic(self, a, b):
        """
        Вычисление эвристической функции (манхэттенское расстояние)
        
        Args:
            a: Первая точка
            b: Вторая точка
            
        Returns:
            int: Манхэттенское расстояние между точками
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, node):
        """
        Получение соседних узлов для текущей позиции
        
        Args:
            node: Текущая позиция
            
        Returns:
            list: Список соседних позиций
        """
        r, c = node
        neighbors = []
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] != CellType.OBSTACLE.value:
                neighbors.append((nr, nc))
        
        return neighbors
    
    def find_path(self, start, goal):
        """
        Поиск пути от начальной до конечной точки
        
        Args:
            start: Начальная позиция
            goal: Конечная позиция
            
        Returns:
            list: Список позиций, составляющих путь
        """
        if start == goal:
            return [start]
        
        open_list = []
        heapq.heappush(open_list, (0, start))
        
        closed_set = set()
        
        g_score = {start: 0}
        
        f_score = {start: self.heuristic(start, goal)}
        
        came_from = {}
        
        while open_list:
            current_f, current = heapq.heappop(open_list)
            
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in [i[1] for i in open_list]:
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        return []
    
    def update_grid(self, changes):
        """
        Обновление сетки карты
        
        Args:
            changes: Список изменений (позиция, новое значение)
        """
        for pos, new_value in changes:
            self.grid[pos] = new_value

# Класс робота
class Robot:
    """Класс для представления робота"""
    
    def __init__(self, start_pos, lidar_range=2, shelf_positions=None):
        """
        Инициализация робота
        
        Args:
            start_pos: Начальная позиция робота
            lidar_range: Дальность действия лидара
            shelf_positions: Список позиций стеллажей
        """
        self.pos = start_pos
        self.start_pos = start_pos
        self.carrying_item = False
        self.path = []
        self.path_index = 0
        self.lidar_range = lidar_range
        self.task_completed = True
        self.pickup_pos = None
        self.dropoff_pos = None
        self.phase = "idle"  # idle, to_pickup, to_dropoff, to_start
        self.move_delay = 0.1  # Задержка между перемещениями (0.5 секунды)
        self.last_move_time = 0
        self.shelf_positions = shelf_positions if shelf_positions is not None else []
    
    def update(self, grid, pathfinder, current_time):
        """
        Обновление состояния робота
        
        Args:
            grid: Сетка карты
            pathfinder: Объект для поиска пути
            current_time: Текущее время
        """
        if current_time - self.last_move_time < self.move_delay:
            return
            
        if not self.path or self.path_index >= len(self.path):
            return
        
        # Проверка на препятствия в диапазоне лидара
        changes = []
        for r in range(max(0, self.pos[0] - self.lidar_range), min(grid.shape[0], self.pos[0] + self.lidar_range + 1)):
            for c in range(max(0, self.pos[1] - self.lidar_range), min(grid.shape[1], self.pos[1] + self.lidar_range + 1)):
                if grid[r, c] != pathfinder.grid[r, c]:
                    changes.append(((r, c), grid[r, c]))
        
        if changes:
            # Обнаружены изменения в области видимости лидара
            pathfinder.update_grid(changes)
            
            # Перепланируем путь в зависимости от текущей фазы
            if self.phase == "to_pickup":
                self.path = pathfinder.find_path(self.pos, self.pickup_pos)
            elif self.phase == "to_dropoff":
                self.path = pathfinder.find_path(self.pos, self.dropoff_pos)
            elif self.phase == "to_start":
                self.path = pathfinder.find_path(self.pos, self.start_pos)
                
            self.path_index = 0
            
            if not self.path or len(self.path) <= 1:
                # Не удалось найти путь, останемся на месте
                self.path = [self.pos]
                return
        
        # Проверяем следующую позицию на пути
        next_pos = self.path[self.path_index]
        
        # Проверяем, не занята ли следующая позиция препятствием
        if grid[next_pos] == CellType.OBSTACLE.value or grid[next_pos] == CellType.MOVING_OBSTACLE.value:
            # Если следующая позиция занята, перепланируем путь
            if self.phase == "to_pickup":
                self.path = pathfinder.find_path(self.pos, self.pickup_pos)
            elif self.phase == "to_dropoff":
                self.path = pathfinder.find_path(self.pos, self.dropoff_pos)
            elif self.phase == "to_start":
                self.path = pathfinder.find_path(self.pos, self.start_pos)
            
            self.path_index = 0
            return
        
        # Перемещение робота
        if self.path_index < len(self.path):
            self.pos = self.path[self.path_index]
            self.path_index += 1
            self.last_move_time = current_time
        
        # Проверка достижения цели
        if self.phase == "to_pickup" and self.pos == self.pickup_pos:
            self.carrying_item = True
            self.phase = "to_dropoff"
            self.path = pathfinder.find_path(self.pos, self.dropoff_pos)
            self.path_index = 0
        elif self.phase == "to_dropoff" and self.pos == self.dropoff_pos:
            self.carrying_item = False
            self.phase = "to_start"
            self.path = pathfinder.find_path(self.pos, self.start_pos)
            self.path_index = 0
        elif self.phase == "to_start" and self.pos == self.start_pos:
            self.task_completed = True
            self.phase = "idle"
            self.path = []
    
    def set_task(self, pickup_pos, dropoff_pos, pathfinder):
        """
        Установка новой задачи для робота
        
        Args:
            pickup_pos: Позиция точки забора
            dropoff_pos: Позиция точки выгрузки
            pathfinder: Объект для поиска пути
        """
        self.pickup_pos = pickup_pos
        self.dropoff_pos = dropoff_pos
        self.task_completed = False
        self.carrying_item = False
        self.phase = "to_pickup"
        
        # Обновляем сетку с учетом статических препятствий
        for pos in self.shelf_positions:
            pathfinder.grid[pos] = CellType.OBSTACLE.value
            
        # Проверяем достижимость точки забора груза
        path_to_pickup = pathfinder.find_path(self.pos, pickup_pos)
        if not path_to_pickup:
            # Если путь к точке забора недостижим, помечаем задачу как выполненную
            self.task_completed = True
            self.phase = "idle"
            return
            
        # Проверяем достижимость точки выгрузки
        path_to_dropoff = pathfinder.find_path(pickup_pos, dropoff_pos)
        if not path_to_dropoff:
            # Если путь к точке выгрузки недостижим, помечаем задачу как выполненную
            self.task_completed = True
            self.phase = "idle"
            return
            
        # Если оба пути достижимы, устанавливаем путь к точке забора
        self.path = path_to_pickup
        self.path_index = 0

# Класс симулятора склада
class WarehouseSimulator:
    """Класс для симуляции работы склада"""
    
    def __init__(self):
        """Инициализация симулятора склада"""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH + INFO_WIDTH, HEIGHT))
        pygame.display.set_caption("Симулятор склада")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.shelf_positions = []
        self.start_pos = (GRID_HEIGHT // 2, GRID_WIDTH // 2)
        self.pickup_pos = None
        # Фиксированная точка выгрузки (в центре склада, но правее)
        self.dropoff_pos = (GRID_HEIGHT // 2, GRID_WIDTH // 2 + 5)
        self.last_obstacle_time = 0
        self.obstacle_interval = 2.0  # Интервал появления препятствий в секундах
        self.font = pygame.font.SysFont('Arial', 16)
        self.info_surface = pygame.Surface((INFO_WIDTH, HEIGHT))
        
        # Генерация начальной карты
        self.generate_warehouse_layout()
        
        # Создаем робота с учетом стеллажей
        self.robot = Robot(self.start_pos, shelf_positions=self.shelf_positions)
        self.pathfinder = Pathfinder(self.grid)
        self.dynamic_obstacles = set()
        self.moving_obstacles = []
        
        # Генерация движущихся препятствий
        self.generate_moving_obstacles(3)
        
        # Генерация первой задачи
        self.generate_new_task()
    
    def generate_warehouse_layout(self):
        """Генерация начальной карты склада"""
        # Создание основных проходов
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                # Границы склада
                if r == 0 or r == GRID_HEIGHT - 1 or c == 0 or c == GRID_WIDTH - 1:
                    self.grid[r, c] = CellType.OBSTACLE.value
        
        # Стеллажи только по бокам карты
        # Левая сторона
        for row in range(3, GRID_HEIGHT - 3, 3):
            for col in [2, 3]:
                self.grid[row][col] = CellType.OBSTACLE.value
                self.grid[row+1][col] = CellType.OBSTACLE.value
                self.shelf_positions.append((row, col))
                self.shelf_positions.append((row+1, col))
        
        # Правая сторона
        for row in range(3, GRID_HEIGHT - 3, 3):
            for col in [GRID_WIDTH - 3, GRID_WIDTH - 4]:
                self.grid[row][col] = CellType.OBSTACLE.value
                self.grid[row+1][col] = CellType.OBSTACLE.value
                self.shelf_positions.append((row, col))
                self.shelf_positions.append((row+1, col))
        
        # Добавляем дополнительные стеллажи в центре
        center_col = GRID_WIDTH // 2
        for row in range(3, GRID_HEIGHT - 3, 3):
            for col in [center_col - 2, center_col - 1, center_col + 1, center_col + 2]:
                self.grid[row][col] = CellType.OBSTACLE.value
                self.grid[row+1][col] = CellType.OBSTACLE.value
                self.shelf_positions.append((row, col))
                self.shelf_positions.append((row+1, col))
    
    def generate_moving_obstacles(self, count):
        """
        Генерация движущихся препятствий
        
        Args:
            count: Количество препятствий для генерации
        """
        for _ in range(count):
            attempts = 0
            while attempts < 20:  # Ограничиваем количество попыток
                r = random.randint(5, GRID_HEIGHT - 5)
                c = random.randint(5, GRID_WIDTH - 5)
                
                # Проверяем, что место свободно
                if (self.grid[r, c] == CellType.EMPTY.value and 
                    (r, c) != self.robot.pos and 
                    (r, c) != self.pickup_pos and 
                    (r, c) != self.dropoff_pos and
                    (r, c) not in self.shelf_positions):
                    
                    # Создаем движущееся препятствие
                    obstacle = MovingObstacle((r, c), self.grid)
                    self.moving_obstacles.append(obstacle)
                    self.grid[r, c] = CellType.MOVING_OBSTACLE.value
                    break
                
                attempts += 1
    
    def generate_new_task(self):
        """Генерация новой задачи для робота"""
        if self.robot.task_completed:
            max_attempts = 10  # Максимальное количество попыток генерации задачи
            attempts = 0
            
            while attempts < max_attempts:
                # Генерируем точку загрузки (выбор случайного стеллажа)
                if self.shelf_positions:
                    # Выбираем случайную позицию рядом со стеллажом
                    shelf_pos = random.choice(self.shelf_positions)
                    r, c = shelf_pos
                    
                    # Выбираем соседнюю проходимую ячейку для точки забора груза
                    possible_pickup_spots = []
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH and self.grid[nr, nc] == CellType.EMPTY.value:
                            possible_pickup_spots.append((nr, nc))
                    
                    if possible_pickup_spots:
                        self.pickup_pos = random.choice(possible_pickup_spots)
                        self.grid[self.pickup_pos] = CellType.PICKUP.value
                        
                        # Задаем задачу роботу с фиксированной точкой выгрузки
                        self.robot.set_task(self.pickup_pos, self.dropoff_pos, self.pathfinder)
                        
                        # Если задача не была помечена как выполненная, значит она достижима
                        if not self.robot.task_completed:
                            break
                
                attempts += 1
                
                # Если все попытки исчерпаны, очищаем точку забора
                if attempts == max_attempts:
                    if self.pickup_pos:
                        self.grid[self.pickup_pos] = CellType.EMPTY.value
                        self.pickup_pos = None
    
    def draw(self):
        """Отрисовка всей сцены"""
        self.screen.fill(LIGHT_BLUE)  # Светло-голубой фон для проходов
        
        # Отрисовка сетки
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if (r, c) == self.robot.pos:
                    # Робот (синий без груза, фиолетовый с грузом)
                    pygame.draw.rect(self.screen, BLUE if not self.robot.carrying_item else PURPLE, rect)
                    # Круг внутри прямоугольника для робота
                    pygame.draw.circle(self.screen, WHITE, 
                                     (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2), 
                                     CELL_SIZE // 3)
                elif self.grid[r, c] == CellType.MOVING_OBSTACLE.value:
                    # Движущиеся препятствия (коричневые)
                    pygame.draw.rect(self.screen, BROWN, rect)
                    # Добавляем "стрелку" для обозначения направления
                    for obstacle in self.moving_obstacles:
                        if obstacle.pos == (r, c):
                            dir_r, dir_c = obstacle.direction
                            center_x = c * CELL_SIZE + CELL_SIZE // 2
                            center_y = r * CELL_SIZE + CELL_SIZE // 2
                            end_x = center_x + dir_c * (CELL_SIZE // 3)
                            end_y = center_y + dir_r * (CELL_SIZE // 3)
                            pygame.draw.line(self.screen, BLACK, (center_x, center_y), (end_x, end_y), 3)
                elif (r, c) in self.dynamic_obstacles:
                    # Динамические препятствия (красные)
                    pygame.draw.rect(self.screen, RED, rect)
                elif (r, c) in self.shelf_positions or (r == 0 or r == GRID_HEIGHT - 1 or c == 0 or c == GRID_WIDTH - 1):
                    # Статические препятствия - стеллажи и границы (серые)
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif (r, c) == self.pickup_pos:
                    # Точка загрузки (зеленая)
                    pygame.draw.rect(self.screen, GREEN, rect)
                    # Значок товара
                    pygame.draw.rect(self.screen, BLACK, 
                                   pygame.Rect(c * CELL_SIZE + 5, r * CELL_SIZE + 5, 
                                             CELL_SIZE - 10, CELL_SIZE - 10), 2)
                elif (r, c) == self.dropoff_pos:
                    # Точка выгрузки (темно-зеленая)
                    pygame.draw.rect(self.screen, DARK_GREEN, rect)
                    # Значок выгрузки
                    pygame.draw.polygon(self.screen, WHITE, [
                        (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + 5),
                        (c * CELL_SIZE + 5, r * CELL_SIZE + CELL_SIZE - 5),
                        (c * CELL_SIZE + CELL_SIZE - 5, r * CELL_SIZE + CELL_SIZE - 5)
                    ])
                elif (r, c) == self.start_pos:
                    # Стартовая позиция (желтая)
                    pygame.draw.rect(self.screen, YELLOW, rect)
                
                # Контур ячейки
                pygame.draw.rect(self.screen, BLACK, rect, 1)
        
        # Отрисовка зоны действия лидара
        lidar_rect = pygame.Rect(
            (self.robot.pos[1] - self.robot.lidar_range) * CELL_SIZE,
            (self.robot.pos[0] - self.robot.lidar_range) * CELL_SIZE,
            (2 * self.robot.lidar_range + 1) * CELL_SIZE,
            (2 * self.robot.lidar_range + 1) * CELL_SIZE
        )
        pygame.draw.rect(self.screen, (255, 255, 255, 128), lidar_rect, 2)
        
        # Отрисовка пути
        if self.robot.path:
            for i in range(len(self.robot.path) - 1):
                if i >= self.robot.path_index - 1:
                    start_pos = self.robot.path[i]
                    end_pos = self.robot.path[i + 1]
                    start_pixel = (start_pos[1] * CELL_SIZE + CELL_SIZE // 2, start_pos[0] * CELL_SIZE + CELL_SIZE // 2)
                    end_pixel = (end_pos[1] * CELL_SIZE + CELL_SIZE // 2, end_pos[0] * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.line(self.screen, ORANGE, start_pixel, end_pixel, 3)
        
        # Отрисовка информационного окна
        self.info_surface.fill(WHITE)
        
        # Заголовок
        title = self.font.render("Информация о симуляции", True, BLACK)
        self.info_surface.blit(title, (10, 10))
        
        # Статус робота
        status_text = f"Фаза: {self.robot.phase}"
        status_surface = self.font.render(status_text, True, BLACK)
        self.info_surface.blit(status_surface, (10, 40))
        
        # Информация о пути
        if self.robot.path:
            path_info = f"Длина пути: {len(self.robot.path)}"
            path_surface = self.font.render(path_info, True, BLACK)
            self.info_surface.blit(path_surface, (10, 60))
            
            pos_info = f"Позиция робота: {self.robot.pos}"
            pos_surface = self.font.render(pos_info, True, BLACK)
            self.info_surface.blit(pos_surface, (10, 80))
            
            if self.robot.phase == "to_pickup":
                target_info = f"Цель (забор груза): {self.robot.pickup_pos}"
                target_surface = self.font.render(target_info, True, GREEN)
                self.info_surface.blit(target_surface, (10, 100))
            elif self.robot.phase == "to_dropoff":
                target_info = f"Цель (доставка груза): {self.robot.dropoff_pos}"
                target_surface = self.font.render(target_info, True, DARK_GREEN)
                self.info_surface.blit(target_surface, (10, 100))
            elif self.robot.phase == "to_start":
                target_info = f"Цель (возврат на базу): {self.robot.start_pos}"
                target_surface = self.font.render(target_info, True, YELLOW)
                self.info_surface.blit(target_surface, (10, 100))
        
        # Счетчик препятствий
        obstacles_info = f"Статические препятствия: {len(self.shelf_positions)}"
        obstacles_surface = self.font.render(obstacles_info, True, BLACK)
        self.info_surface.blit(obstacles_surface, (10, 120))
        
        dynamic_info = f"Динамические препятствия: {len(self.dynamic_obstacles)}"
        dynamic_surface = self.font.render(dynamic_info, True, BLACK)
        self.info_surface.blit(dynamic_surface, (10, 140))
        
        moving_info = f"Движущиеся препятствия: {len(self.moving_obstacles)}"
        moving_surface = self.font.render(moving_info, True, BLACK)
        self.info_surface.blit(moving_surface, (10, 160))
        
        # Легенда (перемещаем выше)
        legend_title = self.font.render("Легенда:", True, BLACK)
        self.info_surface.blit(legend_title, (10, 200))
        
        legend_items = [
            ("Робот", BLUE), 
            ("Робот с грузом", PURPLE),
            ("Стеллаж", GRAY), 
            ("Препятствие", RED),
            ("Движ. объект", BROWN),
            ("Точка загрузки", GREEN), 
            ("Точка выгрузки", DARK_GREEN)
        ]
        
        for i, (text, color) in enumerate(legend_items):
            # Прямоугольник с цветом
            rect = pygame.Rect(10, 220 + i * 25, 15, 15)
            pygame.draw.rect(self.info_surface, color, rect)
            pygame.draw.rect(self.info_surface, BLACK, rect, 1)
            
            # Текст легенды
            legend_text = self.font.render(text, True, BLACK)
            self.info_surface.blit(legend_text, (30, 220 + i * 25))
        
        # Инструкция по управлению (перемещаем в самый низ)
        control_info = "Управление:"
        control_surface = self.font.render(control_info, True, BLACK)
        self.info_surface.blit(control_surface, (10, HEIGHT - 90))
        
        pause_info = "Пробел - пауза"
        pause_surface = self.font.render(pause_info, True, BLACK)
        self.info_surface.blit(pause_surface, (10, HEIGHT - 70))
        
        step_info = "N (в паузе) - шаг вперед"
        step_surface = self.font.render(step_info, True, BLACK)
        self.info_surface.blit(step_surface, (10, HEIGHT - 50))
        
        move_info = "M - добавить движущееся препятствие"
        move_surface = self.font.render(move_info, True, BLACK)
        self.info_surface.blit(move_surface, (10, HEIGHT - 30))
        
        # Отображаем информационное окно
        self.screen.blit(self.info_surface, (WIDTH, 0))
        
        pygame.display.flip()
    
    def update_moving_obstacles(self, current_time):
        """
        Обновление движущихся препятствий
        
        Args:
            current_time: Текущее время
            
        Returns:
            bool: True если хотя бы одно препятствие переместилось
        """
        moved = False
        for obstacle in self.moving_obstacles:
            # Очищаем предыдущую позицию, если препятствие переместилось
            if obstacle.prev_pos != obstacle.pos:
                self.grid[obstacle.prev_pos] = CellType.EMPTY.value
            
            # Проверяем следующую позицию на пути
            next_r = obstacle.pos[0] + obstacle.direction[0]
            next_c = obstacle.pos[1] + obstacle.direction[1]
            next_pos = (next_r, next_c)
            
            # Проверяем, не занята ли следующая позиция
            if (0 <= next_r < GRID_HEIGHT and 0 <= next_c < GRID_WIDTH and
                self.grid[next_pos] == CellType.EMPTY.value and
                next_pos != self.robot.pos and
                next_pos != self.pickup_pos and
                next_pos != self.dropoff_pos and
                next_pos not in self.shelf_positions):
                
                # Обновляем позицию
                if obstacle.update(self.grid, current_time, self.robot.pos, self.pickup_pos, self.dropoff_pos, self.shelf_positions):
                    moved = True
            else:
                # Если следующая позиция занята, выбираем новое направление
                obstacle.direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
                obstacle.current_steps = 0
            
            # Отмечаем новую позицию
            self.grid[obstacle.pos] = CellType.MOVING_OBSTACLE.value
        
        return moved

    def move_dynamic_obstacles(self):
        """Генерация и перемещение динамических препятствий"""
        # Очистка предыдущих динамических препятствий
        for pos in self.dynamic_obstacles:
            if self.grid[pos] == CellType.OBSTACLE.value and pos not in [(0, c) for c in range(GRID_WIDTH)] and \
                pos not in [(GRID_HEIGHT - 1, c) for c in range(GRID_WIDTH)] and \
                pos not in [(r, 0) for r in range(GRID_HEIGHT)] and \
                pos not in [(r, GRID_WIDTH - 1) for r in range(GRID_HEIGHT)] and \
                pos not in self.shelf_positions:
                self.grid[pos] = CellType.EMPTY.value
        
        # Создаем новые динамические препятствия
        self.dynamic_obstacles = set()
        num_obstacles = 15
        
        # Центральная зона
        center_r, center_c = GRID_HEIGHT // 2, GRID_WIDTH // 2
        for _ in range(num_obstacles // 3):
            attempts = 0
            while attempts < 10:
                dr = random.randint(-5, 5)
                dc = random.randint(-5, 5)
                r, c = center_r + dr, center_c + dc
                
                if 2 <= r < GRID_HEIGHT - 2 and 2 <= c < GRID_WIDTH - 2:
                    # Добавляем проверку на точку старта и выгрузки
                    if (r, c) != self.robot.pos and \
                       (r, c) != self.pickup_pos and \
                       (r, c) != self.dropoff_pos and \
                       (r, c) != self.start_pos and \
                       self.grid[r, c] == CellType.EMPTY.value and \
                       (r, c) not in self.shelf_positions:
                        self.grid[r, c] = CellType.OBSTACLE.value
                        self.dynamic_obstacles.add((r, c))
                        break
                attempts += 1
        
        # Остальные зоны
        for _ in range(num_obstacles * 2 // 3):
            attempts = 0
            while attempts < 10:
                r = random.randint(2, GRID_HEIGHT - 3)
                c = random.randint(2, GRID_WIDTH - 3)
                
                # Добавляем проверку на точку старта и выгрузки
                if (r, c) != self.robot.pos and \
                   (r, c) != self.pickup_pos and \
                   (r, c) != self.dropoff_pos and \
                   (r, c) != self.start_pos and \
                   self.grid[r, c] == CellType.EMPTY.value and \
                   (r, c) not in self.shelf_positions and \
                   (r, c) not in self.dynamic_obstacles:
                    self.grid[r, c] = CellType.OBSTACLE.value
                    self.dynamic_obstacles.add((r, c))
                    break
                attempts += 1

    def update(self):
        """Обновление состояния симуляции"""
        if not self.paused:
            current_time = time.time()
            
            # Обновление движущихся препятствий
            self.update_moving_obstacles(current_time)
            
            # Обновление динамических препятствий
            if current_time - self.last_obstacle_time > self.obstacle_interval:
                self.move_dynamic_obstacles()
                self.last_obstacle_time = current_time
            
            # Генерация новой задачи, если робот не занят
            if self.robot.task_completed:
                self.generate_new_task()
            
            # Обновление робота
            self.robot.update(self.grid, self.pathfinder, current_time)
        
        self.draw()
    
    def handle_events(self):
        """Обработка событий пользователя"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_n and self.paused:
                    # Шаг симуляции в режиме паузы
                    current_time = time.time()
                    self.robot.update(self.grid, self.pathfinder, current_time)
                elif event.key == pygame.K_m:
                    # Добавление нового движущегося препятствия при нажатии 'M'
                    self.generate_moving_obstacles(1)
    
    def run(self):
        """Запуск симуляции"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(30)
        
        pygame.quit()

if __name__ == "__main__":
    simulator = WarehouseSimulator()
    simulator.run() 
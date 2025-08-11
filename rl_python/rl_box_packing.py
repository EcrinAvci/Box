import json
import random
from collections import defaultdict

# --- Kutu ve Konteyner Sınıfları ---
class Box:
    def __init__(self, x, y, z, weight):
        self.x = x
        self.y = y
        self.z = z
        self.weight = weight
        self.volume = x * y * z

class PlacedBox(Box):
    def __init__(self, x, y, z, weight, posx, posy, posz, rotation):
        super().__init__(x, y, z, weight)
        self.posx = posx
        self.posy = posy
        self.posz = posz
        self.rotation = rotation

class Container:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.placed = []

    def can_place(self, x, y, z, w, h, d):
        # Sınır kontrolü
        if x + w > self.width or y + h > self.height or z + d > self.depth:
            return False
        # Çakışma kontrolü
        for pb in self.placed:
            if not (x + w <= pb.posx or pb.posx + pb.x <= x or
                    y + h <= pb.posy or pb.posy + pb.y <= y or
                    z + d <= pb.posz or pb.posz + pb.z <= z):
                return False
        return True

    def place_box(self, box, posx, posy, posz, rotation):
        pb = PlacedBox(box.x, box.y, box.z, box.weight, posx, posy, posz, rotation)
        self.placed.append(pb)
        return pb

# --- RL Agent ---
class QLearningAgent:
    def __init__(self, actions, alpha=0.2, gamma=0.95, epsilon=0.3):
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def get_state(self, container, placed_count, total_volume):
        fill = total_volume / (container.width * container.height * container.depth)
        return (round(fill, 2), placed_count)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qs = [self.q[(state, a)] for a in self.actions]
        maxq = max(qs)
        max_actions = [a for a, qv in zip(self.actions, qs) if qv == maxq]
        return random.choice(max_actions)

    def update(self, state, action, reward, next_state):
        best_next = max([self.q[(next_state, a)] for a in self.actions])
        self.q[(state, action)] += self.alpha * (reward + self.gamma * best_next - self.q[(state, action)])

# --- Yardımcılar ---
def apply_rotation(box, rotation):
    # rotation: 0=XYZ, 1=XZY, 2=YXZ, 3=ZYX, 4=ZXY, 5=YZX
    dims = [box.x, box.y, box.z]
    if rotation == 0:
        return Box(dims[0], dims[1], dims[2], box.weight)
    elif rotation == 1:
        return Box(dims[0], dims[2], dims[1], box.weight)
    elif rotation == 2:
        return Box(dims[1], dims[0], dims[2], box.weight)
    elif rotation == 3:
        return Box(dims[2], dims[1], dims[0], box.weight)
    elif rotation == 4:
        return Box(dims[2], dims[0], dims[1], box.weight)
    elif rotation == 5:
        return Box(dims[1], dims[2], dims[0], box.weight)
    else:
        return Box(dims[0], dims[1], dims[2], box.weight)

def reward_fn(pb, container, total_volume, placed_count):
    reward = 2000
    fill = total_volume / (container.width * container.height * container.depth)
    if fill > 0.95: reward += 1000
    elif fill > 0.9: reward += 800
    elif fill > 0.85: reward += 600
    elif fill > 0.8: reward += 400
    elif fill > 0.75: reward += 200
    elif fill > 0.7: reward += 100
    reward += placed_count * 50
    # Kenar bonusu
    if pb.posx <= 2 or pb.posx + pb.x >= container.width - 2: reward += 300
    if pb.posy <= 2 or pb.posy + pb.y >= container.height - 2: reward += 300
    if pb.posz <= 2 or pb.posz + pb.z >= container.depth - 2: reward += 300
    # Düşük koordinat bonusu
    reward -= pb.posz * 5
    reward -= pb.posy * 2
    reward -= pb.posx * 1
    # Kompaktlık
    if pb.posx == 0 or pb.posy == 0 or pb.posz == 0: reward += 150
    return reward

# --- Sınır ve Çakışma Kontrolü ---
def is_inside_container(x, y, z, w, h, d, container_w, container_h, container_d):
    return (0 <= x < container_w and 0 <= y < container_h and 0 <= z < container_d and
            x + w <= container_w and y + h <= container_h and z + d <= container_d)

# --- Ana Eğitim Döngüsü ---
def main():
    with open('../data.json') as f:
        data = json.load(f)
    boxes = [Box(d['X'], d['Y'], d['Z'], d['Weight']) for d in data]
    actions = []
    # Pozisyon gridini ve rotasyonları oluştur
    for x in range(0, 100, 5):
        for y in range(0, 100, 5):
            for z in range(0, 100, 5):
                for r in range(6):
                    actions.append((x, y, z, r))
    agent = QLearningAgent(actions)
    best_fill = 0
    best_placed = 0
    best_result = []

    for episode in range(1, 501):  # 500 episode
        container = Container(100, 100, 100)
        placed_boxes = []
        total_volume = 0
        placed_count = 0
        sorted_boxes = sorted(boxes, key=lambda b: -b.volume)
        for box in sorted_boxes:
            state = agent.get_state(container, placed_count, total_volume)
            action = agent.choose_action(state)
            x, y, z, r = action
            rotated = apply_rotation(box, r)
            # Sınır ve çakışma kontrolü
            if is_inside_container(x, y, z, rotated.x, rotated.y, rotated.z, container.width, container.height, container.depth) and container.can_place(x, y, z, rotated.x, rotated.y, rotated.z):
                pb = container.place_box(rotated, x, y, z, r)
                placed_boxes.append(pb)
                total_volume += rotated.volume
                placed_count += 1
                reward = reward_fn(pb, container, total_volume, placed_count)
            else:
                reward = -1000  # Büyük ceza
            next_state = agent.get_state(container, placed_count, total_volume)
            agent.update(state, action, reward, next_state)
        fill = total_volume / (container.width * container.height * container.depth)
        if fill > best_fill:
            best_fill = fill
            best_placed = placed_count
            # Sadece geçerli kutuları kaydet
            best_result = [{
                'X': pb.x, 'Y': pb.y, 'Z': pb.z,
                'posX': pb.posx, 'posY': pb.posy, 'posZ': pb.posz,
                'Weight': pb.weight, 'Rotation': pb.rotation
            } for pb in placed_boxes if is_inside_container(pb.posx, pb.posy, pb.posz, pb.x, pb.y, pb.z, container.width, container.height, container.depth)]
        if episode % 50 == 0 or episode == 1:
            print(f"Episode {episode}/500 | Yerleşen kutu: {placed_count} | Doluluk: {fill*100:.2f}%")
    # Sonucu kaydet
    with open('rl_placement_result.json', 'w') as f:
        json.dump({'best_fill': best_fill, 'best_placed': best_placed, 'boxes': best_result}, f, indent=2)
    print(f"En iyi RL sonucu: {best_placed} kutu, doluluk: {best_fill*100:.2f}%")

if __name__ == '__main__':
    main()

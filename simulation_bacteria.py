#adaptive time, adding springs with probability

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches

# Disk
R = 1.  # radius


# Linear spring properties
k_s = 20.0  # Spring constant
l = 1.2  # Natural length

# Repulsion force properties
k_c = 10000.0  # Repulsion force constant

# Torsion spring properties
kt_par = 5.0  # Torsion spring constant (parallel component)
kt_bot = 5.0  # Torsion spring constant (bottom component)
theta0 = np.pi  # Natural state angle (radians)

# Initial conditions
initial_positions = [
    np.array([0, 0]),
    np.array([l, 0]),
    np.array([2 * l, 0]),
    np.array([3 * l, 0])
]
initial_velocities = [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])]
initial_ids = [0, 1, 2, 3]  # Particle IDs

# Time step and simulation limits
initial_dt = 0.00025
eps = 1e-6
t_max = 60.0
max_chains = 128  # Upper limit of particle chains to stop the calculation
current_time = 0

# Timing for particle addition
add_particle_interval = 1./0.7  # Add a particle every fixed time
particle_add_interval_counter = 0  # Particle addition interval counter
add_particle_interval_steps = int(add_particle_interval / initial_dt)  # Number of steps for particle addition interval

# Small range to prevent backflow
overlap_threshold = 0.1  # Set velocity to 0 if distance is below 0.1

# Threshold for splitting (total distance between particles)
split_threshold = 10.

# Function to slightly randomize the position of particles
def random_shift(position, scale=0.1):
    return position + np.random.uniform(-scale, scale, size=position.shape)

# Bounding box calculation
def get_bounding_box(chain):
    positions, _, _, _ = chain
    x_min = min(pos[0] for pos in positions) - R
    x_max = max(pos[0] for pos in positions) + R
    y_min = min(pos[1] for pos in positions) - R
    y_max = max(pos[1] for pos in positions) + R
    return x_min, x_max, y_min, y_max

# Bounding box overlap check
def are_bounding_boxes_overlapping(box1, box2):
    x_min1, x_max1, y_min1, y_max1 = box1
    x_min2, x_max2, y_min2, y_max2 = box2
    if x_max1 < x_min2 or x_max2 < x_min1 or y_max1 < y_min2 or y_max2 < y_min1:
        return False
    return True

# repulsion force calculation
def simple_repulsion_force(Xj, Xl, k_c, R):
    r = np.linalg.norm(Xj - Xl) / (2 * R)
    if r < 1:
        f = -k_c / (2 * R ** 2) * (1 - 2 * R / np.linalg.norm(Xj - Xl)) * (Xj - Xl)
        return f
    else:
        return np.array([0.0, 0.0])

# Check and resolve collisions between particle chains, applying repulsion force
def check_and_resolve_chain_collision(chain1, chain2, k_c, R):
    pos1, vel1, ids1, _ = chain1
    pos2, vel2, ids2, _ = chain2

    box1 = get_bounding_box(chain1)
    box2 = get_bounding_box(chain2)

    # Check if bounding boxes overlap
    if are_bounding_boxes_overlapping(box1, box2):
        for _ in range(20):  # Apply repulsion force up to 10 times until overlap is resolved
            overlap_exists = False
            for i, pos1_i in enumerate(pos1):
                for j, pos2_j in enumerate(pos2):
                    if np.linalg.norm(pos1_i - pos2_j) < 2 * R:
                        force = simple_repulsion_force(pos1_i, pos2_j, k_c, R)
                        if np.linalg.norm(force) > 0:
                            overlap_exists = True
                            vel1[i] += force * dt
                            vel2[j] -= force * dt

            if not overlap_exists:
                break  # Exit loop if overlap is resolved

            # Update positions
            for i in range(len(pos1)):
                pos1[i] += vel1[i] * dt
            for j in range(len(pos2)):
                pos2[j] += vel2[j] * dt

# Resolve overlaps between end particles
def resolve_end_particle_overlap(chain1, chain2, k_c, R):
    pos1, vel1, ids1, _ = chain1
    pos2, vel2, ids2, _ = chain2

    for _ in range(20):  # Apply repulsion force up to 10 times
        overlap_exists = False
        for pos1_i, vel1_i in zip(pos1, vel1):
            for pos2_j, vel2_j in zip(pos2, vel2):
                if np.linalg.norm(pos1_i - pos2_j) < 2 * R:
                    force = simple_repulsion_force(pos1_i, pos2_j, k_c, R)
                    if np.linalg.norm(force) > 0:
                        overlap_exists = True
                        vel1_i += force * dt
                        vel2_j -= force * dt

        if not overlap_exists:
            break  # Exit loop if overlap is resolved

        # Update positions
        for i in range(len(pos1)):
            pos1[i] += vel1[i] * dt
        for j in range(len(pos2)):
            pos2[j] += vel2[j] * dt

# Torsion spring force calculation (parallel component)
def torsion_spring_par(k, positions, kt_par, theta0, eps=1e-6):
    V = np.zeros(2)
    n = len(positions)
    
    def norm(vec):
        return np.linalg.norm(vec)

    if k == 0 and n >= 3:
        Xj, Xj1, Xj2 = positions[k], positions[k + 1], positions[k + 2]
        V -= kt_par / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
             ((np.dot(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.cos(theta0)) * \
             ((Xj2 - Xj1) - np.dot(Xj2 - Xj1, Xj - Xj1) * (Xj - Xj1) / (norm(Xj - Xj1) ** 2 + eps))
    elif k == 1:
        Xj_1, Xj = positions[k - 1], positions[k]
        if n >= 3:
            Xj1 = positions[k + 1]
            V += kt_par / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                 ((np.dot(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.cos(theta0)) * \
                 ((Xj_1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj1 - Xj) / (norm(Xj1 - Xj) ** 2 + eps) + \
                  (Xj1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj_1 - Xj) / (norm(Xj_1 - Xj) ** 2 + eps))
        if n > 3:
            Xj2 = positions[k + 2]
            V -= kt_par / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
                 ((np.dot(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.cos(theta0)) * \
                 ((Xj2 - Xj1) - np.dot(Xj2 - Xj1, Xj - Xj1) * (Xj - Xj1) / (norm(Xj - Xj1) ** 2 + eps))
    elif k == n - 1 and n >= 3:
        Xj_2, Xj_1, Xj = positions[k - 2], positions[k - 1], positions[k]
        V -= kt_par / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
             ((np.dot(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.cos(theta0)) * \
             ((Xj_2 - Xj_1) - np.dot(Xj - Xj_1, Xj_2 - Xj_1) * (Xj - Xj_1) / (norm(Xj - Xj_1) ** 2 + eps))
    elif k == n - 2:
        Xj = positions[k]
        if n > 3:
            Xj_2, Xj_1 = positions[k - 2], positions[k - 1]
            V -= kt_par / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
                 ((np.dot(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.cos(theta0)) * \
                 ((Xj_2 - Xj_1) - np.dot(Xj - Xj_1, Xj_2 - Xj_1) * (Xj - Xj_1) / (norm(Xj - Xj_1) ** 2 + eps))
        if n >= 3:
            Xj1 = positions[k + 1]
            Xj_1 = positions[k - 1]
            V += kt_par / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                 ((np.dot(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.cos(theta0)) * \
                 ((Xj_1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj1 - Xj) / (norm(Xj1 - Xj) ** 2 + eps) + \
                  (Xj1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj_1 - Xj) / (norm(Xj_1 - Xj) ** 2 + eps))
    else:
        Xj_2, Xj_1, Xj, Xj1, Xj2 = positions[k - 2], positions[k - 1], positions[k], positions[k + 1], positions[k + 2]
        V -= kt_par / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
             ((np.dot(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.cos(theta0)) * \
             ((Xj_2 - Xj_1) - np.dot(Xj - Xj_1, Xj_2 - Xj_1) * (Xj - Xj_1) / (norm(Xj - Xj_1) ** 2 + eps))
        V += kt_par / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
             ((np.dot(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.cos(theta0)) * \
             ((Xj_1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj1 - Xj) / (norm(Xj1 - Xj) ** 2 + eps) + \
              (Xj1 - Xj) - np.dot(Xj1 - Xj, Xj_1 - Xj) * (Xj_1 - Xj) / (norm(Xj_1 - Xj) ** 2 + eps))
        V -= kt_par / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
             ((np.dot(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.cos(theta0)) * \
             ((Xj2 - Xj1) - np.dot(Xj2 - Xj1, Xj - Xj1) * (Xj - Xj1) / (norm(Xj - Xj1) ** 2 + eps))
    return V

# Torsion spring（bottom component）
def torsion_spring_bot(k, positions, kt_bot, theta0, eps=1e-6):
    V = np.zeros(2)
    n = len(positions)
    
    def norm(vec):
        return np.linalg.norm(vec)

    if k == 0 and n >= 3:
        Xj, Xj1, Xj2 = positions[k], positions[k + 1], positions[k + 2]
        V -= kt_bot / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
             ((np.cross(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.sin(theta0)) * \
             np.array([-(Xj2 - Xj1)[1] + ((Xj - Xj1)[0] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1), (Xj2 - Xj1)[0] + ((Xj - Xj1)[1] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1)])
    elif k == 1:
        Xj, Xj_1 = positions[k], positions[k - 1]
        if n >= 3:
            Xj1 = positions[k + 1]
            V += kt_bot / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                 ((np.cross(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.sin(theta0)) * \
                 (np.array([(Xj_1 - Xj)[1] - (((Xj1 - Xj)[0]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj), -(Xj_1 - Xj)[0] - (((Xj1 - Xj)[1]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj)]) + \
                  np.array([-(Xj1 - Xj)[1] + (((Xj_1 - Xj)[0]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj), (Xj1 - Xj)[0] + (((Xj_1 - Xj)[1]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj)]))
        if n > 3:
            Xj2 = positions[k + 2]
            V -= kt_bot / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
                 ((np.cross(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.sin(theta0)) * \
                 np.array([-(Xj2 - Xj1)[1] + ((Xj - Xj1)[0] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1), (Xj2 - Xj1)[0] + ((Xj - Xj1)[1] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1)])
    elif k == n - 1 and n >= 3:
        Xj_2, Xj_1, Xj = positions[k - 2], positions[k - 1], positions[k]
        V -= kt_bot / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
             ((np.cross(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.sin(theta0)) * \
             np.array([(Xj_2 - Xj_1)[1] - ((Xj - Xj_1)[0]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1), -(Xj_2 - Xj_1)[0] - ((Xj - Xj_1)[1]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1)])
    elif k == n - 2:
        Xj = positions[k]
        if n > 3:
            Xj_2, Xj_1 = positions[k - 2], positions[k - 1]
            V -= kt_bot / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
                 ((np.cross(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.sin(theta0)) * \
                 np.array([(Xj_2 - Xj_1)[1] - ((Xj - Xj_1)[0]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1), -(Xj_2 - Xj_1)[0] - ((Xj - Xj_1)[1]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1)])
        if n >= 3:
            Xj1 = positions[k + 1]
            Xj_1 = positions[k - 1]
            V += kt_bot / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
                 ((np.cross(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.sin(theta0)) * \
                 (np.array([(Xj_1 - Xj)[1] - (((Xj1 - Xj)[0]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj), -(Xj_1 - Xj)[0] - (((Xj1 - Xj)[1]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj)]) + \
                  np.array([-(Xj1 - Xj)[1] + (((Xj_1 - Xj)[0]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj), (Xj1 - Xj)[0] + (((Xj_1 - Xj)[1]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj)]))
    else:
        Xj_2, Xj_1, Xj, Xj1, Xj2 = positions[k - 2], positions[k - 1], positions[k], positions[k + 1], positions[k + 2]
        V -= kt_bot / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) * \
             ((np.cross(Xj - Xj_1, Xj_2 - Xj_1)) / (norm(Xj - Xj_1) * norm(Xj_2 - Xj_1) + eps) - np.sin(theta0)) * \
             np.array([(Xj_2 - Xj_1)[1] - ((Xj - Xj_1)[0]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1), -(Xj_2 - Xj_1)[0] - ((Xj - Xj_1)[1]) / (norm(Xj - Xj_1) ** 2 + eps) * np.cross(Xj - Xj_1, Xj_2 - Xj_1)])
        V += kt_bot / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) * \
             ((np.cross(Xj1 - Xj, Xj_1 - Xj)) / (norm(Xj1 - Xj) * norm(Xj_1 - Xj) + eps) - np.sin(theta0)) * \
             (np.array([(Xj_1 - Xj)[1] - (((Xj1 - Xj)[0]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj), -(Xj_1 - Xj)[0] - (((Xj1 - Xj)[1]) / (norm(Xj1 - Xj) ** 2 + eps)) * np.cross(Xj1 - Xj, Xj_1 - Xj)]) + \
              np.array([-(Xj1 - Xj)[1] + (((Xj_1 - Xj)[0]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj), (Xj1 - Xj)[0] + (((Xj_1 - Xj)[1]) / (norm(Xj_1 - Xj) ** 2 + eps)) * np.cross(Xj_1 - Xj, Xj1 - Xj)]))
        V -= kt_bot / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) * \
             ((np.cross(Xj2 - Xj1, Xj - Xj1)) / (norm(Xj2 - Xj1) * norm(Xj - Xj1) + eps) - np.sin(theta0)) * \
             np.array([-(Xj2 - Xj1)[1] + ((Xj - Xj1)[0] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1), (Xj2 - Xj1)[0] + ((Xj - Xj1)[1] / (norm(Xj - Xj1) ** 2 + eps)) * np.cross(Xj - Xj1, Xj2 - Xj1)])
    return V

# coefficient for adhesion spring
k_adhesion = 1.  # stiffness of adhesion spring
adhesion_break_threshold = 1.0  # Detachment threshold
lambda_poisson=0.6

# adhesion force
def adhesion_force(position, k_adhesion, ground_y=0.0):
    ground_position = np.array([position[0], ground_y])
    distance_vector = ground_position - position
    force = k_adhesion * distance_vector
    return force, np.linalg.norm(force)

def get_num_adhesive_springs(particle_id):

    return 1


# adhesion points
adhesion_points = {}

adhesion_points.clear()

def apply_adhesion_forces(chain, k_adhesion, adhesion_break_threshold):
    positions, velocities, ids, _ = chain
    forces = [np.zeros(2) for _ in positions]
    
    # Apply adhesion force to the first and last particles
    for i in [0, len(positions) - 1]:
        if ids[i] not in adhesion_points:
            continue  # Skip if there is no spring for that particle

        remaining_springs = []
        for ground_position in adhesion_points[ids[i]]:
            force, force_magnitude = adhesion_force(positions[i], k_adhesion, ground_y=ground_position[1])
            
            if force_magnitude <= adhesion_break_threshold:
                forces[i] += force
                remaining_springs.append(ground_position)  # Keep the springs that are not broken
            else:
                print(f"Adhesion spring at particle {ids[i]} broke due to excessive force.")
        
        # Keep only the springs that are not broken
        adhesion_points[ids[i]] = remaining_springs

        # If all springs are broken, remove this particle from adhesion_points
        if not adhesion_points[ids[i]]:
            del adhesion_points[ids[i]]

    return forces


# For recording
all_positions = []
all_times = []  # Record the time at each step
particle_chains = [(initial_positions, initial_velocities, initial_ids, None)]  # None initializes the split position
add_particle_counters = [0]  # Maintain the addition interval counter for each particle chain


# List of colors
chain_colors = ['blue', 'yellow', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'black', 'pink', 'navy', 'lightblue']
next_color_index = len(particle_chains)  # Index to use for assigning new colors

# Colors corresponding to the initial particle chains
particle_chain_colors = [chain_colors[i % len(chain_colors)] for i in range(len(particle_chains))]

# List to track the colors of particle chains at each step
all_chain_colors = []

# Set the range of positions

def calculate_bounds(all_positions):
    x_min = min(pos[0] for frame in all_positions for chain in frame for pos in chain[0])
    x_max = max(pos[0] for frame in all_positions for chain in frame for pos in chain[0])
    y_min = min(pos[1] for frame in all_positions for chain in frame for pos in chain[0])
    y_max = max(pos[1] for frame in all_positions for chain in frame for pos in chain[0])
    return x_min, x_max, y_min, y_max


adhesion_points_history = []

try:
    while current_time < t_max and len(particle_chains) <= max_chains:
        new_chains = []
        new_chain_colors = []
        new_add_particle_counters = []
        current_positions = []

        for chain_idx, (positions, velocities, ids, split_index) in enumerate(particle_chains):
            forces = [np.zeros(2) for _ in positions]

            # Linear Spring and Torsion Spring force
            for i in range(len(positions) - 1):
                distance = positions[i + 1] - positions[i]
                distance_norm = np.linalg.norm(distance)
                force_spring = k_s * (distance_norm - l) * (distance / distance_norm)

                forces[i] += force_spring
                forces[i + 1] -= force_spring

            for i in range(len(forces)):
                forces[i] += torsion_spring_par(i, positions, kt_par, theta0)
                forces[i] += torsion_spring_bot(i, positions, kt_bot, theta0)

            # Resolve overlaps
            for i in range(len(positions) - 1):
                distance = positions[i + 1] - positions[i]
                if np.linalg.norm(distance) < overlap_threshold:
                    velocities[i] = np.array([0, 0])
                    velocities[i + 1] = np.array([0, 0])

            # Probabilistically add adhesion springs
            # Add springs via a Poisson process only to the first and last particles
            if current_time > 0:  # Do not add springs in the initial state
                for i in [0, len(positions) - 1]:  # Only the first and last particles
                    p_add_spring = 1 - np.exp(-lambda_poisson * dt)
                    if np.random.rand() < p_add_spring:
                        # Save the current x-y position of the particle as a fixed point
                        if ids[i] not in adhesion_points:
                            adhesion_points[ids[i]] = []
                        adhesion_points[ids[i]].append(positions[i].copy())  # Add the particle's position as a fixed point
                        print(f"Adhesion spring added to particle {i} (ID: {ids[i]}) at time {current_time}")
                        
            adhesion_points_history.append(adhesion_points.copy())

            # Calculate adhesion force and add to forces
            adhesion_forces = apply_adhesion_forces((positions, velocities, ids, split_index),
                                                    k_adhesion, adhesion_break_threshold)
            forces = [f + adhesion_f for f, adhesion_f in zip(forces, adhesion_forces)]
            
            
            # Dynamic adjustment of time step
            forces_magnitudes = [np.linalg.norm(f) for f in forces]
            max_force = max(forces_magnitudes) if forces_magnitudes else 0
            dt = 0.7 * min(initial_dt, R / max(eps, max_force))

            for i in range(len(positions)):
                velocities[i] = forces[i]   # In overdamped systems, forces are treated directly as velocities
                positions[i] = positions[i] + velocities[i] * dt

            
            current_positions.append((positions.copy(), ids.copy()))

            # division
            total_distance = sum(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))
            if total_distance > split_threshold:
                half = len(positions) // 2
                if len(positions) % 2 == 0:
                    chain1_positions = positions[:half]
                    chain1_velocities = velocities[:half]
                    chain1_ids = ids[:half]
                    chain2_positions = positions[half:]
                    chain2_velocities = velocities[half:]
                    chain2_ids = ids[half:]
                    split_index_chain1 = half - 1
                    split_index_chain2 = 0
                else:
                    chain1_positions = positions[:half]
                    chain1_velocities = velocities[:half]
                    chain1_ids = ids[:half]
                    chain2_positions = positions[half:]
                    chain2_velocities = velocities[half:]
                    chain2_ids = ids[half:]
                    split_index_chain1 = half - 1
                    split_index_chain2 = 0

                    if len(chain1_positions) > 1:
                        direction_vector = chain1_positions[-1] - chain1_positions[-2]
                    else:
                        direction_vector = np.array([1, 0])  # Default direction
                    direction = direction_vector / np.linalg.norm(direction_vector)
                    distance = 1.5
                    new_position = chain1_positions[-1] + distance * direction
                    new_velocity = np.array([0, 0], dtype=np.float64)
                    new_id = max(chain1_ids + chain2_ids) + 1
                    chain1_positions.append(new_position)
                    chain1_velocities.append(new_velocity)
                    chain1_ids.append(new_id)

                # Apply a slight random shift
                chain1_positions = [random_shift(pos) for pos in chain1_positions]
                chain2_positions = [random_shift(pos) for pos in chain2_positions]

                print(f"Time {current_time}: Split occurred.")
                print(f"Chain 1: {chain1_positions}")
                print(f"Chain 2: {chain2_positions}")

                # Resolve overlaps between the end particles of the split chains
                resolve_end_particle_overlap((chain1_positions, chain1_velocities, chain1_ids, split_index_chain1),
                                             (chain2_positions, chain2_velocities, chain2_ids, split_index_chain2),
                                             k_c, R)

                new_chains.append((chain1_positions, chain1_velocities, chain1_ids, current_time))
                new_chains.append((chain2_positions, chain2_velocities, chain2_ids, current_time))

                # Inherit the original color
                new_chain_colors.append(particle_chain_colors[chain_idx])
                new_chain_colors.append(chain_colors[next_color_index % len(chain_colors)])
                next_color_index += 1

                # Also inherit the particle addition interval counter
                new_add_particle_counters.append(add_particle_counters[chain_idx])
                new_add_particle_counters.append(add_particle_counters[chain_idx])

                # Output split information to the log
                print(f"Time {current_time}: Chain {chain_idx} split. Chain 1 color: {new_chain_colors[-2]}, Chain 2 color: {new_chain_colors[-1]}")
            else:
                new_chains.append((positions, velocities, ids, split_index))
                new_chain_colors.append(particle_chain_colors[chain_idx])
                new_add_particle_counters.append(add_particle_counters[chain_idx])

            # Add particles
            if new_add_particle_counters[-1] >= add_particle_interval_steps:
                new_id = max(ids) + 1
                new_velocity = np.array([0, 0], dtype=np.float64)

                if len(positions) > 1:
                    # Randomly select a position between two particles
                    insert_index = np.random.randint(0, len(positions) - 1)
                    new_position = (positions[insert_index] + positions[insert_index + 1]) / 2
                    positions.insert(insert_index + 1, new_position)
                    velocities.insert(insert_index + 1, new_velocity)
                    ids.insert(insert_index + 1, new_id)

                    print(f"Time {current_time}: Particle added between particle {insert_index} and {insert_index + 1} at {new_position} with ID {new_id}.")
                    new_add_particle_counters[-1] = 0
            else:
                new_add_particle_counters[-1] += 1

        # Check and resolve collisions between particle chains, applying repulsion force
        for i in range(len(new_chains)):
            for j in range(i + 1, len(new_chains)):
                check_and_resolve_chain_collision(new_chains[i], new_chains[j], k_c, R)

        all_positions.append(current_positions)
        all_times.append(current_time)
        all_chain_colors.append(particle_chain_colors.copy())
        particle_chains = new_chains
        particle_chain_colors = new_chain_colors
        add_particle_counters = new_add_particle_counters

        current_time += dt

except IndexError as e:
    print(f"Simulation stopped due to an error: {e}")

finally:

    max_adhesive_springs = 10
    
    tolerance_factor = 0.005
    desired_frame_count = 500
    # Save the animation whether the simulation ends normally or due to an error
    print("Saving animation...")
    frame_indices = np.linspace(0, len(all_positions) - 1, num=desired_frame_count).astype(int)
    all_positions_sampled = [all_positions[i] for i in frame_indices]
    all_chain_colors_sampled = [all_chain_colors[i] for i in frame_indices]
    all_times_sampled = [all_times[i] for i in frame_indices]
    
        # Calculate the range of positions
    x_min, x_max, y_min, y_max = calculate_bounds(all_positions_sampled)

    # Animation setup and saving
    fig, ax = plt.subplots()
    ax.set_xlim(x_min - 2 * R, x_max + 2 * R)
    ax.set_ylim(y_min - 2 * R, y_max + 2 * R)

    # Function to sample adhesion points history
    def sample_adhesion_points_history(adhesion_points_history, total_frames):
        return [adhesion_points_history[int(i)] for i in np.linspace(0, len(adhesion_points_history) - 1, total_frames).astype(int)]

    # Function to get color from a colormap
    def get_color_from_cmap(value, max_value, cmap_name='viridis'):
        cmap = plt.get_cmap(cmap_name)
        normalized_value = value / max_value  # Normalize the value to a range of 0 to 1
        color = cmap(normalized_value)
        print(f"Adhesive Color for value {value}/{max_value}: {color}")  # Output the color for debugging
        return color

    # Initialization function
    def init():
        return []

    # Frame update function
    def update(frame_idx):
        frame = all_positions_sampled[frame_idx]
        colors = all_chain_colors_sampled[frame_idx]
        sim_time = all_times_sampled[frame_idx]
        ax.clear()
        ax.set_xlim(x_min - 2 * R, x_max + 2 * R)
        ax.set_ylim(y_min - 2 * R, y_max + 2 * R)
        elements = []

        # Use sampled data
        current_adhesion_points = sampled_adhesion_points_history[frame_idx]
        print(f"Frame {frame_idx}: Adhesion points at time {sim_time}: {current_adhesion_points}")

        for chain_idx, (positions, ids) in enumerate(frame):
            for i, pos in enumerate(positions):
                # Draw the original particles
                disk = patches.Circle(pos, R, color=colors[chain_idx])
                ax.add_patch(disk)
                elements.append(disk)

                # Apply color to particles with adhesion springs
                num_adhesive_springs = len(current_adhesion_points.get(ids[i], []))
                if num_adhesive_springs > 0 and i in [0, len(positions) - 1]:  # Only the first and last particles
                    adhesive_color = get_color_from_cmap(num_adhesive_springs, max_adhesive_springs)
                    print(f"Particle ID {ids[i]}: Number of Adhesive Springs: {num_adhesive_springs}, Color: {adhesive_color}")  # For debugging
                    adhesive_circle = patches.Circle(pos, R * 0.5, color=adhesive_color, alpha=0.8)
                    ax.add_patch(adhesive_circle)
                    elements.append(adhesive_circle)

            # Draw springs
            for i in range(len(positions) - 1):
                distance = np.linalg.norm(positions[i + 1] - positions[i])
                tolerance = tolerance_factor * l
                spring_color = 'red' if distance > l + tolerance else 'gray' if distance < l - tolerance else 'black'
                spring, = ax.plot([positions[i][0], positions[i + 1][0]],
                                  [positions[i][1], positions[i + 1][1]],
                                  color=spring_color)
                elements.append(spring)

        time_text = ax.text(0.05, 0.95, f'Time: {sim_time:.2f} s', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        elements.append(time_text)

        return elements

    # Animation setup and execution
    fig, ax = plt.subplots()

    # Sampling adhesion points history
    total_frames = len(all_positions_sampled)
    sampled_adhesion_points_history = sample_adhesion_points_history(adhesion_points_history, total_frames)

    print(f"Total frames in animation: {total_frames}")
    print(f"Total entries in sampled_adhesion_points_history: {len(sampled_adhesion_points_history)}")

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=20)
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Disks Connected by Springs (with Repulsion) and Adding New Particles')
    plt.gca().set_aspect('equal', adjustable='box')

    # Save the animation
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('particle_simulation_by_chain_poisson_adhesion.mp4', writer=writer)

    plt.show()

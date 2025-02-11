import taichi as ti

# Initialize Taichi
ti.init(arch=ti.cpu)

# Parameters
num_particles = 100  # Number of particles in the rope
grid_size = 64  # Size of the simulation grid
gravity = -9.81  # Gravity acceleration
time_step = 0.005  # Time step for simulation
particle_mass = 0.1  # Mass of each particle
rest_length = 0.01  # Rest length for spring forces in the rope
stiffness = 100.0  # Stiffness of the rope
damping = 5.5  # Damping factor for the rope
friction_coefficient = 0.8  # Coefficient of friction between rope and box

# Rigid box attributes
box_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
box_size = (0.1, 0.05)  # Width, height of the box
box_velocity = ti.Vector.field(2, dtype=ti.f32, shape=())

# Particle attributes
positions = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
grid_velocities = ti.Vector.field(2, dtype=ti.f32, shape=(grid_size, grid_size))
grid_mass = ti.field(dtype=ti.f32, shape=(grid_size, grid_size))

# Initialize the box position
@ti.kernel
def initialize_box():
    box_pos[None] = [0.5, 0.3]  # Initial position of the box
    box_velocity[None] = [0.0, -0.01]  # Velocity of the box (moving down)

@ti.kernel
def initialize():
    for i in range(num_particles):
        positions[i] = [0.5 + i * rest_length, 0.5]  # Initialize the rope positions
        velocities[i] = [0.0, 0.0]  # Initialize velocities to zero

@ti.kernel
def update():
    # Clear grid
    grid_mass.fill(0)
    grid_velocities.fill(0)

    # Step 1: Particle to grid
    for i in range(num_particles):
        # Get the grid cell
        gx = int(positions[i][0] * grid_size)
        gy = int(positions[i][1] * grid_size)

        # Add particle mass to grid
        grid_mass[gx, gy] += particle_mass
        grid_velocities[gx, gy] += velocities[i] * particle_mass

    # Step 2: Compute grid velocity
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_mass[i, j] > 0:
                grid_velocities[i, j] /= grid_mass[i, j]  # Average velocity

    # Step 3: Update velocities with grid velocities
    for i in range(num_particles):
        gx = int(positions[i][0] * grid_size)
        gy = int(positions[i][1] * grid_size)

        # Get grid velocity
        velocities[i] += grid_velocities[gx, gy] * time_step

        # Apply gravity
        velocities[i][1] += gravity * time_step

        # Collision with the ground (y = 0)
        if positions[i][1] < 0:
            positions[i][1] = 0
            velocities[i][1] = 0

            # Apply simple collision response
            normal_velocity = velocities[i].dot(ti.Vector([0, 1]))
            if normal_velocity < 0:
                # Reflect the velocity along the normal
                velocities[i] -= normal_velocity * ti.Vector([0, 1])

    # Step 4: Apply spring forces between particles for stiffness and damping
    for i in range(1, num_particles):
        # Calculate the distance between particles
        diff = positions[i] - positions[i - 1]
        distance = diff.norm()
        if distance > rest_length:
            # Stiffness force
            force = stiffness * (distance - rest_length)
            direction = diff.normalized()
            velocities[i] -= direction * force * time_step / particle_mass
            velocities[i - 1] += direction * force * time_step / particle_mass

            # Damping force
            rel_velocity = velocities[i] - velocities[i - 1]
            damping_force = damping * rel_velocity.dot(direction)
            velocities[i] -= direction * damping_force * time_step / particle_mass
            velocities[i - 1] += direction * damping_force * time_step / particle_mass

    # Step 5: Update positions after calculating all forces
    for i in range(num_particles):
        positions[i] += velocities[i] * time_step

    # Update the box position
    box_pos[None] += box_velocity[None] * time_step

    # Box collision with the rope
    box_x_min = box_pos[None][0] - box_size[0] / 2
    box_x_max = box_pos[None][0] + box_size[0] / 2
    box_y_min = box_pos[None][1] - box_size[1] / 2
    box_y_max = box_pos[None][1] + box_size[1] / 2

    for i in range(num_particles):
        if (positions[i][0] > box_x_min and positions[i][0] < box_x_max and
            positions[i][1] > box_y_min and positions[i][1] < box_y_max):
            # Apply force to the rope particle based on box velocity
            velocities[i] += box_velocity[None] * 0.5  # Adjust strength as needed

            # Apply friction
            friction_force = -friction_coefficient * particle_mass * gravity * velocities[i].normalized()
            velocities[i] += friction_force * time_step / particle_mass  # Apply frictional force

# Create a GUI
gui = ti.GUI('Rope on Tabletop with Pushing Box', (800, 800))

initialize()
initialize_box()

# Main loop
while gui.running:
    update()
    
    # Render particles from top view
    gui.clear()

    # Draw particles (rope)
    for i in range(num_particles):
        gui.circle(positions[i].to_numpy(), radius=5, color=0xFFFFFF)

    # Draw the rigid box
    gui.rect(box_pos[None].to_numpy() - box_size, box_pos[None].to_numpy() + box_size, color=0xFF0000)  # Red color for box

    gui.show()

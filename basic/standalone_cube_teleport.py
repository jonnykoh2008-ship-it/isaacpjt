from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World
stage = omni.usd.get_context().get_stage()              # 3. Stage

initial_position = np.array([0.0, 0.0, 0.5])
teleport_position = np.array([0.0, 0.0, 1.0])

cube_prim = DynamicCuboid(                              # 4. Prim
    prim_path="/World/BlueCube",
    name="blue_cube",
    position=initial_position,
    scale=np.array([0.15, 0.15, 0.15]),
    color=np.array([0.0, 0.0, 1.0]),
)

world.scene.add_default_ground_plane()                  # 5. Scene
world.scene.add(cube_prim)
world.reset()

step_count = 0
was_playing = False

while simulation_app.is_running():                      # 6. Simulation
    if world.is_playing and not was_playing:
        step_count = 0
        cube_prim.set_world_pose(position=initial_position)
        world.reset()
        print("Reset to start: step_count = 0")
        print(f"step: {step_count}")

    was_playing = world.is_playing

    if not world.is_playing:
        time.sleep(0.01)
        continue

    if step_count == 0:
        print(f"step: {step_count}")

    world.step(render=True)
    time.sleep(0.01)
    step_count += 1
    if step_count % 100 == 0:
        print(f"step: {step_count}")
    if step_count == 300:
        cube_prim.set_world_pose(position=teleport_position)
        print("Teleport: cube moved to z=1.0")
    if step_count == 500:
        simulation_app.close()

simulation_app.close()

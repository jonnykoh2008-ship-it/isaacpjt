from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World
stage = omni.usd.get_context().get_stage()              # 3. Stage

cube_prim = DynamicCuboid(                              # 4. Prim
    prim_path="/World/BlueCube",
    name="blue_cube",
    position=np.array([0.0, 0.0, 0.15]),
    scale=np.array([0.3, 0.3, 0.3]),
    color=np.array([0.0, 0.0, 1.0]),
)
cube_two = DynamicCuboid(                              # 4. Prim
    prim_path="/World/RedCube",
    name="red_cube",
    position=np.array([0.0, 0.0,1.0]),
    scale=np.array([0.1, 0.1, 0.1]),
    color=np.array([1.0, 0.0, 0.0]),
)
# cube_three = DynamicCuboid(                              # 4. Prim
#     prim_path="/World/GreenCube",
#     name="green_cube",
#     position=np.array([-1.0, 0.0, 1]),
#     scale=np.array([0.15, 0.15, 0.15]),
#     color=np.array([0.0, 1.0, 0.0]),
# )
world.scene.add_default_ground_plane()                  # 5. Scene
for i in [cube_prim, cube_two]: 
    world.scene.add(i)

world.reset()

step_count = 0
while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)
    time.sleep(0.01)
    step_count += 1
    if step_count % 100 == 0:
        print(f"step: {step_count}")
    if step_count  == 300:
        world.reset()
        cube_two.position=np.array([0, 0, 1.0])
        print("[월드 리셋] 큐브 순간이동")
    # if step_count == 500:
    #     simulation_app.close()

simulation_app.close()

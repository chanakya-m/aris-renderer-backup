from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from aris.config.object_config import ObjectConfig
from aris.config.scene_config import SceneConfig


@dataclass
class RenderConfig:
    scale: float
    spp: int
    block_size: int
    scene: SceneConfig
    integrator: ObjectConfig
    gui: bool
    device: str
    save_exr: bool
    ckpt: str = "outputs/2025-11-02/23-40-59/checkpoints/step_4000.ckpt"

cs = ConfigStore.instance()
cs.store(name="render_schema", node=RenderConfig)

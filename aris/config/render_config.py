from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from aris.config.object_config import ObjectConfig
from aris.config.scene_config import SceneConfig

from typing import Optional


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
    checkpoint: Optional[str] = None

cs = ConfigStore.instance()
cs.store(name="render_schema", node=RenderConfig)

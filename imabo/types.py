from typing import Any, TypeAlias

import numpy as np

ArmKey: TypeAlias = tuple[Any, ...]
ArmConfig: TypeAlias = dict[str, Any]
OptunaConfigs: TypeAlias = dict[str, np.ndarray]

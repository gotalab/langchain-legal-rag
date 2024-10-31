from pathlib import Path
import yaml
from src.core.config import RAGConfig

def load_config(config_path: Path | str | None = None) -> RAGConfig:
    """
    設定ファイルを読み込む
    Args:
        config_path: 設定ファイルのパス（Noneの場合はデフォルト設定を使用）
    Returns:
        RAGConfig: 設定オブジェクト
    """
    if config_path is None:
        return RAGConfig()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    return RAGConfig.model_validate(config_dict) 
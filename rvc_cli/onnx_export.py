# --- rvc_cli/onnx_export.py ---
import os
import logging
import traceback
# Используем функцию из оригинального модуля
from infer.modules.onnx.export import export_onnx as export_onnx_core

logger = logging.getLogger(__name__)

def export_onnx_cli(ckpt_path, onnx_path, config):
    """CLI wrapper for ONNX export."""
    logger.info(f"Exporting {ckpt_path} to {onnx_path}")
    # Убедимся, что директория для вывода существует
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    success = False
    try:
        result = export_onnx_core(ckpt_path, onnx_path)
        logger.info(f"ONNX export result: {result}")
        if "Finished" in result:
            success = True
    except ImportError as e:
        logger.error(f"ONNX export dependencies not found ({e}). Please ensure necessary libraries (e.g., onnx, onnxsim) are installed.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        logger.error(traceback.format_exc())
    return success
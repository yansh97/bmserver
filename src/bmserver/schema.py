from pydantic import BaseModel


class Environment(BaseModel):
    nvidia_device_name: str
    nvidia_device_count: int
    nvidia_driver_version: str
    torch_version: str
    transformers_version: str
    vllm_version: str
    bmserver_version: str

    @classmethod
    def detect(cls) -> "Environment":
        import pynvml
        import torch
        import transformers
        import vllm

        import bmserver

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        nvidia_device_names: list[str] = [
            torch.cuda.get_device_name(device=i)
            for i in range(torch.cuda.device_count())
        ]
        if len(set(nvidia_device_names)) > 1:
            raise RuntimeError("Multiple NVIDIA GPU devices are not supported.")
        pynvml.nvmlInit()
        nvidia_driver_version: str = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        return cls(
            nvidia_device_name=nvidia_device_names[0],
            nvidia_device_count=len(nvidia_device_names),
            nvidia_driver_version=nvidia_driver_version,
            torch_version=str(object=torch.__version__),
            transformers_version=transformers.__version__,
            vllm_version=vllm.__version__,
            bmserver_version=bmserver.__version__,
        )

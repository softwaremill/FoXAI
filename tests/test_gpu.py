import torch

def test_gpu():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() > 0
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
    print(torch.__version__)


if __name__ == "__main__":
    test_gpu()

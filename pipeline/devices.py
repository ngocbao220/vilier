import warnings


def resolve_auto_device(torch_module, requested: str | int | None, *, warn_label: str = "device") -> str:
    requested_text = str(requested if requested is not None else "auto").strip().lower()
    if requested_text == "gpu":
        requested_text = "cuda"
    if requested_text == "auto":
        cuda = getattr(torch_module, "cuda", None)
        return "cuda" if getattr(cuda, "is_available", lambda: False)() else "cpu"
    if requested_text == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        if getattr(cuda, "is_available", lambda: False)():
            return "cuda"
        warnings.warn(f"Requested {warn_label}=cuda but CUDA is not available; falling back to CPU.", RuntimeWarning)
        return "cpu"
    if requested_text.startswith("cuda:"):
        cuda = getattr(torch_module, "cuda", None)
        if getattr(cuda, "is_available", lambda: False)():
            return requested_text
        warnings.warn(f"Requested {warn_label}={requested_text} but CUDA is not available; falling back to CPU.", RuntimeWarning)
        return "cpu"
    return requested_text

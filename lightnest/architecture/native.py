"""Native WGT building blocks; complex application hosts select torch explicitly."""
from pathlib import Path
import weightlang as wgt


def load_recurrent(input_size, hidden_size, *, backend="native", **options):
    """Create an initialized, untrained shared GRU with explicit hidden ports.

    Native execution needs an existing wgtime service (WGTIME_ENDPOINT). The
    same model can use backend='torch'. This does not translate entire LightNest
    mask/top-k/inheritance programs into native execution.
    """
    if any(type(n) is not int or n < 1 for n in (input_size, hidden_size)):
        raise ValueError("GRU widths must be positive integers")
    return wgt.load(Path(__file__).with_name("wgt") / "shared_gru" / "model.wgt",
                    backend=backend, dimensions={"I": input_size, "H": hidden_size}, **options)

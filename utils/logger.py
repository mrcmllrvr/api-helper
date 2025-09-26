import logging 
from time import perf_counter
import functools
from typing import Callable, Any


def log_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()
        values = func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time

        logging.info(f"Execution of {func.__name__}: {run_time:.2f} seconds", stacklevel=2)

        return values

    return wrapper


def log_init():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    with open("logging.log", "a") as log_file:
        log_file.write("\n" + "-" * 80 + "\n")

    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename="logging.log"
                        ) 
    
    return logging.getLogger(__name__)
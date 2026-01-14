import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Basically gives the rank of a particular parrallel thread
    # of the processing of this function
    process_id = tl.program_id(axis=0)

    # Gives the indices of which a particular "process"
    # should be working on
    block_start = process_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Returns a boolean mask, tells the memory loader
    # to ignore the elements that aren't actually relevant
    # if there are more elements in a block than are actually
    # part of the input
    mask = offsets < n_elements

    # loads the values from memory, and simply adds them
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    # takes that output, and puts it back into memory
    # with that same mask
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # preallocate an empty tensor
    output = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    n_elements = output.numel()

    # First, cdiv stands for ceiling division
    # concretely:
    #   regular division: 250 / 64 ≈ 3.9
    #   integer division: 250 // 64 = 3
    #   ceiling division: 250 cdiv 64 = 4
    # So given meta block size = 4, it's just a grid (kind of like jax)
    # where it's just a (4, ) shaped grid as a tuple.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)  # pyrefly:ignore

    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(  # pyrefly:ignore
            lambda: x + y, quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(  # pyrefly:ignore
            lambda: add(x, y), quantiles=quantiles
        )

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)  # pyrefly:ignore


if __name__ == "__main__":
    size = 1024
    benchmark.run(print_data=True)

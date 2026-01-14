import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@torch.compile
def naive_compiled_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    # What num_stages does, at least in the context of tl.range, is it creates
    # the "preloading factor", so if you pass into two, while the first loop
    # is iterating, the second one is loading.
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Input row stride is how many elements are per row
        # so this just finds the start of each row being calculated
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # These are the offsets, within the row, but only upto the block size
        col_offsets = tl.arange(0, BLOCK_SIZE)

        # So for each row assigned to this process
        # They process upto block size
        input_ptrs = row_start_ptr + col_offsets

        # Create the mask
        mask = col_offsets < n_cols

        # Load the row from memory
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        # Basic calculation
        row_minux_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minux_max)
        denominator = tl.sum(numerator)

        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets

        tl.store(output_ptrs, softmax_output, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)  # pyrefly:ignore
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
kernels = {}


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows: int
    n_cols: int
    n_rows, n_cols = x.shape

    # Since the kernel doesn't support multiple blocks for the actual
    # softmax operation, you need to make sure BLOCK_SIZE is greater
    # than the number of columns in the input matrix
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(n_cols)

    # this value is a heuristic, I apparently will find a more robust way
    # to calculate this value later
    num_warps: int = 16

    num_stages: int = 4 if SIZE_SMEM > 200000 else 2

    output: torch.Tensor = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel: triton.compiler.CompiledKernel = softmax_kernel.warmup(  # pyrefly:ignore
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,  # these are warmup exclusive
        grid=(1,),  # these are warmup exclusive
    )
    kernel._init_handles()

    n_regs: int = kernel.n_regs
    size_smem: int = kernel.metadata.shared  # pyrefly:ignore

    # I feel like this is bad naming convention, but that's just how it is for right now
    # NUM_REGS is taken from the GPU, it's now many registers each multiprocessor actually has
    # So what's happening here, is the total number of registers is getting divided among the
    # n_regs, number of registeres needed per thread, by the compile kernel
    # WARP_SIZE, which is how many threads per warp are on my GPU
    # and num_warps, the magic number for how many warps I should use per process/program

    # the final result, occupancy, being the number of programs that can run simulatanuousy
    # on the SMs (streaming multiprocessors)
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    # this next step says, based on registers, I can run this many programs
    # and pased on smem I can run this memory, I need the realistic one
    occupancy = min(occupancy, SIZE_SMEM // size_smem)

    # now that you now how many programs can run per SM, just calculate the total number
    # of programs, this one is pretty self-explanatory
    num_programs = NUM_SM * occupancy

    # now this isn't a GPU constraint, but a size of the matrix constraint
    # you don't need to run more programs than there are rows to calculate
    # the softmax for
    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
    )

    return output


def softmax_with_warps(x: torch.Tensor, num_warps: int) -> torch.Tensor:
    n_rows: int
    n_cols: int
    n_rows, n_cols = x.shape

    # Since the kernel doesn't support multiple blocks for the actual
    # softmax operation, you need to make sure BLOCK_SIZE is greater
    # than the number of columns in the input matrix
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(n_cols)

    # this value is a heuristic, I apparently will find a more robust way
    # to calculate this value later
    num_stages: int = 4 if SIZE_SMEM > 200000 else 2

    output: torch.Tensor = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel: triton.compiler.CompiledKernel = softmax_kernel.warmup(  # pyrefly:ignore
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,  # these are warmup exclusive
        grid=(1,),  # these are warmup exclusive
    )
    kernel._init_handles()

    n_regs: int = kernel.n_regs
    size_smem: int = kernel.metadata.shared  # pyrefly:ignore

    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)

    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
    )

    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["1", "2", "4", "8", "16", "32"],
        line_names=[
            "1 Warp",
            "2 Warps",
            "4 Warps",
            "8 Warps",
            "16 Warps",
            "32 Warps",
        ],  # label name for the lines
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("red", "-"),
            ("purple", "-"),
            ("orange", "-"),
            ("cyan", "-"),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="warp-testing",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function argument not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "1":
        ms = triton.testing.do_bench(lambda: softmax_with_warps(x, num_warps=1))
    if provider == "2":
        ms = triton.testing.do_bench(lambda: softmax_with_warps(x, num_warps=2))
    if provider == "4":
        ms = triton.testing.do_bench(lambda: softmax_with_warps(x, num_warps=4))
    if provider == "8":
        ms = triton.testing.do_bench(lambda: softmax_with_warps(x, num_warps=8))
    if provider == "16":
        ms = triton.testing.do_bench(lambda: softmax_with_warps(x, num_warps=16))
    if provider == "32":
        ms = triton.testing.do_bench(lambda: softmax_with_warps(x, num_warps=32))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=["N"],
#         x_vals=[128 * i for i in range(2, 100)],
#         line_arg="provider",
#         line_vals=["triton", "torch", "naive_softmax", "naive_compiled_softmax"],
#         line_names=[
#             "Triton",
#             "Torch",
#             "Naive Softmax",
#             "Naive Compiled Softmax",
#         ],  # label name for the lines
#         styles=[
#             ("blue", "-"),
#             ("green", "-"),
#             ("red", "-"),
#             ("purple", "-"),
#         ],  # line styles
#         ylabel="GB/s",  # label name for the y-axis
#         plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
#         args={"M": 4096},  # values for function argument not in `x_names` and `y_name`
#     )
# )
# def benchmark(M, N, provider):
#     x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
#     stream = getattr(torch, DEVICE.type).Stream()
#     getattr(torch, DEVICE.type).set_stream(stream)
#     if provider == "torch":
#         ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
#     if provider == "triton":
#         ms = triton.testing.do_bench(lambda: softmax(x))
#     if provider == "naive_softmax":
#         ms = triton.testing.do_bench(lambda: naive_softmax(x))
#     if provider == "naive_compiled_softmax":
#         ms = triton.testing.do_bench(lambda: naive_compiled_softmax(x))
#     gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
#     return gbps(ms)


if __name__ == "__main__":
    benchmark.run(save_path="bench")

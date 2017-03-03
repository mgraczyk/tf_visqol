import numpy as np

def windowed_overlap_add(input_signal, window, overlap, block_proc_func):
  result = np.zeros(input_signal.shape, dtype=input_signal.dtype)
  n = input_signal.shape[1]
  block_size = window.size

  block_start = 0
  while block_start < n:
    block_end = min(block_start + block_size, n)
    this_block_sz = block_end - block_start

    input_block = np.zeros((input_signal.shape[0], block_size), dtype=input_signal.dtype)
    input_block[:, :this_block_sz] = input_signal[:, block_start:block_end]
    output_block = block_proc_func(input_block)
    windowed_output_block = window * output_block
    result[:, block_start:block_end] += windowed_output_block[:, :this_block_sz]

    block_start += overlap
  return result

def main():
  np.random.seed(1)
  n = 32
  x = (1e-3 * np.random.randn(n)).reshape(1, n)
  block_size = 16
  window = np.hamming(block_size + 1)[:block_size]
  overlap = block_size // 2
  output = windowed_overlap_add(x, window, overlap, lambda block: block)

  # TODO(mgraczyk): Pad so that the input and output are the same.
  diff = np.sum(np.abs(output - x))


if __name__ == "__main__":
  main()

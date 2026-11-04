# Tesseract vs PyTorch LSTM Parameter Difference

When comparing the number of parameters between Tesseract's C++ LSTM implementation and PyTorch's `nn.LSTM` module for the exactly same architecture, you will notice a slight discrepancy in the parameter count.

For example, on the `lao` model:
- **Tesseract:** `930,298` weights
- **PyTorch:** `932,858` weights

Difference: **2,560 extra parameters** in PyTorch.

## Where do the extra parameters come from?

This difference comes entirely from how PyTorch implements LSTMs compared to how Tesseract implements them. The extra parameters are **biases**, not weights.

In **PyTorch**, an `nn.LSTM` with `bias=True` maintains **two sets of biases** for each gate:
1. `b_ih` (bias added to the input-to-hidden matrix multiplication)
2. `b_hh` (bias added to the hidden-to-hidden matrix multiplication)

Since an LSTM has 4 internal gates, PyTorch creates $4 \times 2 = 8$ bias values per hidden unit. 
For an LSTM layer with hidden size $H$, PyTorch uses **$8 \times H$ bias parameters**.

In **Tesseract**, mathematically, you only actually need **one combined bias** per gate ($W_{ih}x + W_{hh}h + b$), so Tesseract only uses **$4 \times H$ bias parameters**.

This means PyTorch has exactly $4 \times H$ extra bias parameters per LSTM layer.

### Breakdown of the 2,560 difference:
Let's calculate the extra parameters in the model `[1,48,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c90]`:

1. **SumLSTM(16, 64):** Extra parameters = $4 \times 64 = 256$
2. **FwdLSTM(64, 96):** Extra parameters = $4 \times 96 = 384$
3. **RevLSTM(96, 96):** Extra parameters = $4 \times 96 = 384$
4. **FwdLSTM(96, 384):** Extra parameters = $4 \times 384 = 1536$

Summing these up gives exactly the difference observed:
`256 + 384 + 384 + 1536 = 2,560 extra parameters`.

Adding this to Tesseract's count:
`930,298 (Tesseract) + 2,560 (PyTorch Extra Biases) = 932,858 (PyTorch model count)`.

## Does this affect training or inference?

**No. It is mathematically equivalent and has absolutely no effect on accuracy or performance.**

During the forward pass, PyTorch computes:
`(W_ih * x + b_ih) + (W_hh * h + b_hh)`

This mathematically simplifies to:
`W_ih * x + W_hh * h + (b_ih + b_hh)`

Which is exactly the same formula Tesseract uses:
`(W_ih * x + W_hh * h + b)`

When a weight transfer script moves Tesseract weights into PyTorch, it simply maps the Tesseract bias vector directly into PyTorch's `b_ih`, and initializes the extra `b_hh` to `0`. The formula behaves completely identically to the Tesseract C++ backend.

import os
import re
import struct
import numpy as np
import torch


class _TessReader:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def remaining(self):
        return len(self.data) - self.offset

    def read_bytes(self, n):
        if self.offset + n > len(self.data):
            raise ValueError(f"Unexpected EOF while reading {n} bytes")
        out = self.data[self.offset:self.offset + n]
        self.offset += n
        return out

    def read_int8(self):
        return struct.unpack('<b', self.read_bytes(1))[0]

    def read_uint8(self):
        return struct.unpack('<B', self.read_bytes(1))[0]

    def read_int32(self):
        return struct.unpack('<i', self.read_bytes(4))[0]

    def read_int64(self):
        return struct.unpack('<q', self.read_bytes(8))[0]

    def read_uint32(self):
        return struct.unpack('<I', self.read_bytes(4))[0]

    def read_double(self):
        return struct.unpack('<d', self.read_bytes(8))[0]

    def read_string(self):
        n = self.read_uint32()
        if n < 0:
            raise ValueError("Invalid string length")
        return self.read_bytes(n).decode('utf-8', errors='replace') if n else ''


def _read_network_header(rd):
    _ = rd.read_int8()
    type_name = rd.read_string()
    training_state = rd.read_int8()
    needs_backprop = rd.read_int8()
    network_flags = rd.read_int32()
    ni = rd.read_int32()
    no = rd.read_int32()
    num_weights = rd.read_int32()
    name_len = rd.read_uint32()
    name = rd.read_bytes(name_len).decode('utf-8', errors='replace') if name_len else ''
    return {
        'type_name': type_name,
        'training_state': training_state,
        'needs_backprop': needs_backprop,
        'network_flags': network_flags,
        'ni': ni,
        'no': no,
        'num_weights': num_weights,
        'name': name,
    }


def _read_weight_matrix(rd):
    mode = rd.read_uint8()
    if mode != 128:
        raise ValueError(f"Unsupported weight matrix mode: {mode}")
    rows = rd.read_int32()
    cols = rd.read_int32()
    _ = rd.read_double()  # empty_ sentinel
    n_vals = rows * cols
    raw = rd.read_bytes(n_vals * 8)
    arr = np.frombuffer(raw, dtype=np.float64).reshape(rows, cols)
    w = arr[:, :-1]
    b = arr[:, -1]
    return w, b


def _parse_layers_from_spec(net_spec):
    spec = net_spec.strip().strip('[]')
    tokens = spec.split()

    layers = []
    in_c, cur_h = 1, 48
    is_seq, rnn_in = False, 0
    layer_idx = 0

    for tok in tokens:
        if re.match(r'^\d+,\d+,\d+,\d+$', tok):
            parts = tok.split(',')
            in_c, cur_h = int(parts[3]), int(parts[1])
            continue

        m = re.match(r'^C([ts]?)(\d+),(\d+),(\d+)$', tok)
        if m:
            md, ky, kx, oc = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            layers.append({
                'type': 'conv', 'in_c': in_c, 'out_c': oc,
                'ky': ky, 'kx': kx, 'nonlin': md or 'r', 'name': f'{layer_idx}'
            })
            layer_idx += 2  # conv + activation
            in_c = oc
            continue

        m = re.match(r'^Mp(\d+),(\d+)$', tok)
        if m:
            py, px = int(m.group(1)), int(m.group(2))
            layers.append({'type': 'pool', 'in_c': in_c, 'py': py, 'px': px, 'name': f'{layer_idx}'})
            layer_idx += 1
            cur_h = max(1, cur_h // py)
            continue

        m = re.match(r'^L([frbm])([xy])([s]?)(\d+)$', tok)
        if m:
            d, _ax, sm, hid = m.group(1), m.group(2), m.group(3), int(m.group(4))
            if sm == 's':
                layers.append({'type': 'lstm', 'dir': 'sum', 'in_c': in_c, 'hid': hid, 'name': f'{layer_idx}'})
                layer_idx += 1
                is_seq, rnn_in, in_c = True, hid, hid
                continue

            if not is_seq:
                layers.append({'type': 'collapse', 'in_c': in_c, 'cur_h': cur_h, 'name': f'{layer_idx}'})
                layer_idx += 1
                rnn_in, is_seq = in_c * cur_h, True
                in_c = rnn_in

            inp = rnn_in or in_c
            if d == 'b':
                layers.append({'type': 'lstm', 'dir': 'bi', 'in_c': inp, 'hid': hid, 'name': f'{layer_idx}'})
                layer_idx += 1
                rnn_in, in_c = hid * 2, hid * 2
            elif d == 'r':
                layers.append({'type': 'lstm', 'dir': 'rev', 'in_c': inp, 'hid': hid, 'name': f'{layer_idx}'})
                layer_idx += 1
                rnn_in, in_c = hid, hid
            else:
                layers.append({'type': 'lstm', 'dir': 'fwd', 'in_c': inp, 'hid': hid, 'name': f'{layer_idx}'})
                layer_idx += 1
                rnn_in, in_c = hid, hid
            continue

        m = re.match(r'^O(\d+)c(.+)$', tok)
        if m:
            if not is_seq:
                layers.append({'type': 'collapse', 'in_c': in_c, 'cur_h': cur_h, 'name': f'{layer_idx}'})
                layer_idx += 1
                rnn_in, is_seq = in_c * cur_h, True
                in_c = rnn_in
            layers.append({'type': 'output', 'in_c': rnn_in or in_c, 'name': f'{layer_idx}'})
            layer_idx += 1

    return layers


def _gate_order_tess_to_torch(arr, hidden):
    # Tesseract: [G, I, F, O] -> PyTorch: [I, F, G, O]
    g = arr[0:hidden]
    i = arr[hidden:2 * hidden]
    f = arr[2 * hidden:3 * hidden]
    o = arr[3 * hidden:4 * hidden]
    return np.concatenate([i, f, g, o], axis=0)


def _read_single_lstm(rd, in_c, hid):
    _ = _read_network_header(rd)
    _ = rd.read_int32()  # na = in_c + hid

    w_ih_parts = []
    w_hh_parts = []
    b_parts = []

    for _ in range(4):
        gw, gb = _read_weight_matrix(rd)
        if gw.shape[0] != hid:
            raise ValueError(f"Unexpected LSTM gate rows: got {gw.shape[0]}, expected {hid}")
        if gw.shape[1] < in_c + hid:
            raise ValueError(f"Unexpected LSTM gate cols: got {gw.shape[1]}, expected >= {in_c + hid}")

        w_ih_parts.append(gw[:, :in_c])
        w_hh_parts.append(gw[:, in_c:in_c + hid])
        b_parts.append(gb)

    w_ih_tess = np.concatenate(w_ih_parts, axis=0)
    w_hh_tess = np.concatenate(w_hh_parts, axis=0)
    b_tess = np.concatenate(b_parts, axis=0)

    w_ih = _gate_order_tess_to_torch(w_ih_tess, hid)
    w_hh = _gate_order_tess_to_torch(w_hh_tess, hid)

    # Tesseract stores only combined bias. PyTorch uses bias_ih + bias_hh.
    bias_ih = _gate_order_tess_to_torch(b_tess, hid)
    bias_hh = np.zeros_like(bias_ih)

    return {
        'weight_ih_l0': torch.from_numpy(w_ih.astype(np.float32)),
        'weight_hh_l0': torch.from_numpy(w_hh.astype(np.float32)),
        'bias_ih_l0': torch.from_numpy(bias_ih.astype(np.float32)),
        'bias_hh_l0': torch.from_numpy(bias_hh.astype(np.float32)),
    }


def _remap_output_tess_to_pytorch(weight, bias):
    if weight.shape[0] < 2:
        return weight, bias
    out_w = np.copy(weight)
    out_b = np.copy(bias)
    out_w[0] = weight[1]
    out_b[0] = bias[1]
    out_w[1] = weight[0]
    out_b[1] = bias[0]
    if weight.shape[0] > 2:
        out_w[2:] = weight[2:]
        out_b[2:] = bias[2:]
    return out_w, out_b


def parse_lstm_blob_to_state_dict(lstm_blob, net_spec):
    rd = _TessReader(lstm_blob)
    layers = _parse_layers_from_spec(net_spec)

    series_header = _read_network_header(rd)
    if series_header['type_name'] != 'Series':
        raise ValueError(f"Expected Series network, got {series_header['type_name']}")

    _ = rd.read_int32()  # series sublayer count

    _ = _read_network_header(rd)  # Input
    for _ in range(5):
        rd.read_int32()  # StaticShape fields

    state = {}

    for lc in layers:
        if lc['type'] == 'conv':
            _ = _read_network_header(rd)  # Convolve
            rd.read_int32()  # half_x
            rd.read_int32()  # half_y
            _ = _read_network_header(rd)  # Relu / Tanh

            w2d, b = _read_weight_matrix(rd)
            out_c = w2d.shape[0]
            expected_in = lc['in_c'] * lc['ky'] * lc['kx']
            if w2d.shape[1] != expected_in:
                raise ValueError(
                    f"Conv shape mismatch at layer {lc['name']}: got {w2d.shape[1]}, expected {expected_in}"
                )

            w_perm = w2d.reshape(out_c, lc['kx'], lc['ky'], lc['in_c'])
            w_pt = w_perm.transpose(0, 3, 2, 1)

            state[f"{lc['name']}.weight"] = torch.from_numpy(w_pt.astype(np.float32))
            state[f"{lc['name']}.bias"] = torch.from_numpy(b.astype(np.float32))

        elif lc['type'] == 'pool':
            _ = _read_network_header(rd)
            rd.read_int32()  # x_scale
            rd.read_int32()  # y_scale

        elif lc['type'] == 'collapse':
            _ = _read_network_header(rd)

        elif lc['type'] == 'lstm':
            base = f"{lc['name']}.lstm"
            if lc['dir'] == 'sum' or lc['dir'] == 'fwd':
                lstm_state = _read_single_lstm(rd, lc['in_c'], lc['hid'])
                for k, v in lstm_state.items():
                    state[f"{base}.{k}"] = v

            elif lc['dir'] == 'rev':
                _ = _read_network_header(rd)  # RTLReversed
                _ = rd.read_int32()  # wrapped sublayers
                lstm_state = _read_single_lstm(rd, lc['in_c'], lc['hid'])
                for k, v in lstm_state.items():
                    state[f"{base}.{k}"] = v

            elif lc['dir'] == 'bi':
                _ = _read_network_header(rd)  # Parallel
                _ = rd.read_int32()  # branch count

                fwd_state = _read_single_lstm(rd, lc['in_c'], lc['hid'])
                for k, v in fwd_state.items():
                    state[f"{base}.{k}"] = v

                _ = _read_network_header(rd)  # RTLReversed
                _ = rd.read_int32()  # wrapped sublayers
                rev_state = _read_single_lstm(rd, lc['in_c'], lc['hid'])
                for k, v in rev_state.items():
                    rev_key = k.replace('_l0', '_l0_reverse')
                    state[f"{base}.{rev_key}"] = v
            else:
                raise ValueError(f"Unsupported LSTM direction: {lc['dir']}")

        elif lc['type'] == 'output':
            _ = _read_network_header(rd)
            w, b = _read_weight_matrix(rd)
            w, b = _remap_output_tess_to_pytorch(w, b)
            state[f"{lc['name']}.weight"] = torch.from_numpy(w.astype(np.float32))
            state[f"{lc['name']}.bias"] = torch.from_numpy(b.astype(np.float32))

    metadata = {}
    if rd.remaining() > 4:
        try:
            metadata['exported_net_spec'] = rd.read_string()
            if rd.remaining() >= 4:
                metadata['training_flags'] = rd.read_int32()
            if rd.remaining() >= 4:
                metadata['iteration'] = rd.read_int32()
            if rd.remaining() >= 4:
                metadata['sample_iteration'] = rd.read_int32()
            if rd.remaining() >= 4:
                metadata['null_char'] = rd.read_int32()
        except Exception:
            # Older/non-exported blobs may not include trailing metadata.
            pass

    return state, metadata


def _extract_component(tessdata_bytes, component_index):
    if len(tessdata_bytes) < 4:
        raise ValueError("Invalid .traineddata file")

    n_entries = struct.unpack_from('<i', tessdata_bytes, 0)[0]
    if n_entries <= 0:
        raise ValueError("Invalid tessdata header")

    offsets = []
    pos = 4
    for _ in range(n_entries):
        offsets.append(struct.unpack_from('<q', tessdata_bytes, pos)[0])
        pos += 8

    if component_index >= len(offsets):
        raise ValueError(f"Component {component_index} not present")

    start = offsets[component_index]
    if start < 0:
        raise ValueError(f"Component {component_index} is missing")

    candidates = [o for o in offsets if o > start]
    end = min(candidates) if candidates else len(tessdata_bytes)
    return tessdata_bytes[start:end]


def _load_lstm_blob_from_path(path):
    with open(path, 'rb') as f:
        raw = f.read()

    lower = path.lower()
    if lower.endswith('.lstm'):
        return raw
    if lower.endswith('.traineddata'):
        return _extract_component(raw, 17)

    # Try as .traineddata first, then fallback to raw .lstm bytes.
    try:
        return _extract_component(raw, 17)
    except Exception:
        return raw


def load_tesseract_pretrained_state_dict(path, net_spec):
    if not path:
        raise ValueError("Empty Tesseract model path")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    lstm_blob = _load_lstm_blob_from_path(path)
    state, metadata = parse_lstm_blob_to_state_dict(lstm_blob, net_spec)
    return state, metadata


def apply_compatible_pretrained_weights(model, pretrained_state):
    model_state = model.state_dict()
    loaded = []
    skipped = []

    for key, value in pretrained_state.items():
        if key not in model_state:
            skipped.append((key, 'missing'))
            continue
        if model_state[key].shape != value.shape:
            skipped.append((key, f"shape {tuple(value.shape)} != {tuple(model_state[key].shape)}"))
            continue

        model_state[key] = value.to(dtype=model_state[key].dtype)
        loaded.append(key)

    model.load_state_dict(model_state, strict=False)
    return loaded, skipped

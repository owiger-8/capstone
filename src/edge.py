import os
import numpy as np
import onnxruntime as ort
import hnswlib


class EdgePredictor:
    def __init__(self, onnx_path="cnn_gru_base.onnx", scaler_path="scaler.npy",
                 latent_dim=32, memory_path_prefix=None):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.mn, self.mx = np.load(scaler_path)
        self.index = hnswlib.Index(space="l2", dim=latent_dim)
        self.index.init_index(max_elements=500000, ef_construction=200, M=16)
        self.values = []
        self.count = 0
        self.memory_path_prefix = memory_path_prefix
        if memory_path_prefix:
            self.load_memory(memory_path_prefix)

    def _scale(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mn) / (self.mx - self.mn + 1e-8)

    def _inv_scale(self, x):
        return x * (self.mx - self.mn + 1e-8) + self.mn

    def predict(self, window, y_curr=None):
        x = self._scale(np.asarray(window, dtype=np.float32)).reshape(1, 60, 1).astype(np.float32)
        base_pred, latent = self.session.run(None, {self.input_name: x})
        base_pred = float(base_pred[0, 0])
        latent = latent[0].astype(np.float32)
        if self.count >= 10:
            labels, _ = self.index.knn_query(latent.reshape(1, -1), k=5)
            mem_pred = float(np.mean([self.values[i] for i in labels[0]]))
            final_scaled = 0.6 * base_pred + 0.4 * mem_pred
        else:
            final_scaled = base_pred
        self.index.add_items(latent.reshape(1, -1), [self.count])
        self.values.append(float(self._scale(y_curr)) if y_curr is not None else float(base_pred))
        self.count += 1
        return float(self._inv_scale(final_scaled)), latent

    def save_memory(self, path_prefix=None):
        """Persist the adapted HNSW memory index and values to disk."""
        path_prefix = path_prefix or self.memory_path_prefix
        if not path_prefix:
            return False
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        self.index.save_index(path_prefix + ".index")
        np.save(path_prefix + "_values.npy", np.array(self.values, dtype=np.float32))
        return True

    def load_memory(self, path_prefix=None):
        """Load a previously persisted HNSW memory index and values from disk."""
        path_prefix = path_prefix or self.memory_path_prefix
        if not path_prefix:
            return False
        index_path = path_prefix + ".index"
        values_path = path_prefix + "_values.npy"
        if os.path.exists(index_path) and os.path.exists(values_path):
            self.index.load_index(index_path, max_elements=500000)
            self.values = np.load(values_path).tolist()
            self.count = len(self.values)
            return True
        return False

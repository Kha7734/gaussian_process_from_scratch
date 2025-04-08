import numpy as np
from kernels.rbf_kernel import rbf_kernel

LOWER_BOUND_LENGTH_SCALE = 0.5
NUM_EPOCHS = 500

class HyperparameterOptimizer:
    def __init__(self, gp_model, lr=0.001):
        self.gp_model = gp_model
        self.lr = lr
        # Thêm biến để theo dõi quá trình tối ưu
        self.best_mll = -np.inf
        self.best_params = None

    def compute_mll(self, X, y, params=None):
        """Compute the Marginal Log Likelihood (MLL)"""
        if params is not None:
            length_scale, variance, noise = params
        else:
            length_scale = self.gp_model.length_scale
            variance = self.gp_model.variance
            noise = self.gp_model.noise

        # Đảm bảo các tham số dương và hợp lý
        if length_scale <= 0 or variance <= 0 or noise <= 0:
            return -np.inf

        try:
            # Tính kernel matrix với jitter để đảm bảo ổn định số học
            K = rbf_kernel(X, X, length_scale, variance)
            K_noise = K + noise * np.eye(len(X))
            
            # Sử dụng SVD thay vì Cholesky để tăng tính ổn định
            # hoặc giữ Cholesky nhưng thêm jitter lớn hơn
            jitter = 1e-6
            while True:
                try:
                    L = np.linalg.cholesky(K_noise + jitter * np.eye(len(X)))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
                    if jitter > 1.0:  # Nếu jitter quá lớn, có vấn đề với dữ liệu
                        return -np.inf
            
            # Tính MLL
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            mll = -0.5 * y.T @ alpha
            mll -= np.sum(np.log(np.diagonal(L)))
            mll -= 0.5 * len(X) * np.log(2 * np.pi)
            
            return float(mll)  # Đảm bảo trả về scalar
        except Exception as e:
            print(f"Error in compute_mll: {e}")
            return -np.inf

    def step(self, X, y):
        """Thực hiện một bước tối ưu hóa"""
        # Lấy tham số hiện tại
        current_params = np.array([
            # self.gp_model.length_scale,
            max(self.gp_model.length_scale, LOWER_BOUND_LENGTH_SCALE),
            self.gp_model.variance,
            self.gp_model.noise
        ])
        
        # Tính MLL hiện tại
        current_mll = self.compute_mll(X, y)
        
        # Sử dụng log-space để đảm bảo tham số luôn dương
        log_params = np.log(current_params)
        
        # Tính gradient theo phương pháp finite difference
        grads = np.zeros_like(log_params)
        eps = 1e-4  # Epsilon nhỏ hơn để tính gradient chính xác hơn
        
        for i in range(len(log_params)):
            # Tăng tham số trong log-space
            log_params_plus = log_params.copy()
            log_params_plus[i] += eps
            
            # Chuyển về không gian thực
            params_plus = np.exp(log_params_plus)
            
            # Tính MLL với tham số mới
            mll_plus = self.compute_mll(X, y, params=params_plus)
            
            # Tính gradient
            grads[i] = (mll_plus - current_mll) / eps
        
        # Cập nhật tham số trong log-space
        new_log_params = log_params + self.lr * grads
        
        # Chuyển về không gian thực và cập nhật model
        new_params = np.exp(new_log_params)
        # Đảm bảo length_scale không quá nhỏ
        new_params[0] = max(new_params[0], LOWER_BOUND_LENGTH_SCALE)
        
        # Kiểm tra xem tham số mới có cải thiện MLL không
        new_mll = self.compute_mll(X, y, params=new_params)
        
        # Nếu MLL cải thiện hoặc là lần đầu tiên, cập nhật tham số
        if new_mll > self.best_mll:
            self.best_mll = new_mll
            self.best_params = new_params
            
            # Cập nhật model
            self.gp_model.length_scale = new_params[0]
            self.gp_model.variance = new_params[1]
            self.gp_model.noise = new_params[2]
            
            return True  # Đã cải thiện
        
        return False  # Không cải thiện

    def optimize(self, X, y, epochs=NUM_EPOCHS):
        """Train hyperparameters"""
        self.best_mll = -np.inf
        self.best_params = None
        
        history = []
        no_improvement_count = 0
        
        for epoch in range(epochs):
            improved = self.step(X, y)
            current_mll = self.compute_mll(X, y)
            history.append(current_mll)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: MLL = {current_mll:.4f}, "
                      f"length_scale = {self.gp_model.length_scale:.4f}, "
                      f"variance = {self.gp_model.variance:.4f}, "
                      f"noise = {self.gp_model.noise:.6f}")
            
            # Early stopping nếu không cải thiện
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= 10:  # Dừng sau 10 epochs không cải thiện
                    print(f"Early stopping at epoch {epoch+1}: No improvement for 10 epochs")
                    break
            else:
                no_improvement_count = 0
        
        # Đảm bảo sử dụng tham số tốt nhất
        if self.best_params is not None:
            self.gp_model.length_scale = self.best_params[0]
            self.gp_model.variance = self.best_params[1]
            self.gp_model.noise = self.best_params[2]
        
        return {
            'length_scale': self.gp_model.length_scale,
            'variance': self.gp_model.variance,
            'noise': self.gp_model.noise,
            'history': history
        }

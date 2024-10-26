from skCCM import CCM
import numpy as np

class ExtendedCCM(CCM):
    """Extended Convergent Cross Mapping for detecting time-delayed causal interactions."""
    
    def __init__(self, time_series_x, time_series_y, max_lag=50, max_dim=4, tau=1):
        """Initialize ExtendedCCM with time series data and parameters."""
        super().__init__(weights='exp')
        
        if len(time_series_x) == 0 or len(time_series_y) == 0:
            raise ValueError("Time series cannot be empty")
            
        self.time_series_x = time_series_x
        self.time_series_y = time_series_y
        self.max_lag = max_lag
        self.max_dim = max_dim
        self.tau = tau

    def embed_with_lag(self, time_series, lag, embed_dim):
        """
        Create time-lagged embedding following Sugihara's implementation.
        For positive lags: variable predicting future values
        For negative lags: variable predicting past values
        """
        if embed_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
            
        N = len(time_series)
        embedding = []
        
        # For positive lags: predict future from past
        # For negative lags: predict past from future
        if lag >= 0:
            # Starting from earlier times predicting forward
            for i in range(N - (embed_dim-1)*lag):
                point = [time_series[i + j*lag] for j in range(embed_dim)]
                embedding.append(point)
        else:
            # Starting from later times predicting backward
            neg_lag = abs(lag)
            for i in range(N - (embed_dim-1)*neg_lag):
                point = [time_series[N-1 - (i + j*neg_lag)] for j in range(embed_dim)]
                embedding.append(point)
        
        if not embedding:
            raise ValueError("No valid embedding points generated")
            
        return np.array(embedding)

    def compute_cross_map_skill(self, x1, x2, lag, embed_dim, library_length):
        """Compute cross-map skill for a given lag using CCM methods."""
        X1_embed = self.embed_with_lag(x1, lag, embed_dim)
        X2_embed = self.embed_with_lag(x2, lag, embed_dim)

        min_length = min(len(X1_embed), len(X2_embed))
        X1_embed = X1_embed[:min_length]
        X2_embed = X2_embed[:min_length]

        train_size = min(min_length, library_length)
        X1_train = X1_embed[:train_size]
        X2_train = X2_embed[:train_size]
        X1_test = X1_embed[-train_size:]
        X2_test = X2_embed[-train_size:]

        super().fit(X1_train, X2_train)
        X1_pred, X2_pred = super().predict(X1_test, X2_test, [train_size])
        score_x, score_y = super().score()
        
        return score_x[0], score_y[0]

    def detect_causality(self, x1, x2, lags, embed_dim=3, library_length=200):
        """Detect causality type based on cross-map skill patterns."""
        if embed_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
            
        ccm_skill = []
        
        for lag in lags:
            try:
                score_x, score_y = self.compute_cross_map_skill(
                    x1, x2, lag, embed_dim, library_length
                )
                ccm_skill.append((lag, score_x, score_y))
            except ValueError as e:
                print(f"Warning: Skipping lag {lag}: {str(e)}")
                continue
        
        if not ccm_skill:
            raise ValueError("No valid results obtained for any lag value")
            
        ccm_skill = np.array(ccm_skill)
        
        # Find optimal lags and corresponding skills
        x_to_y_idx = np.argmax(ccm_skill[:, 1])
        y_to_x_idx = np.argmax(ccm_skill[:, 2])
        
        opt_lag_x_to_y = ccm_skill[x_to_y_idx, 0]
        opt_lag_y_to_x = ccm_skill[y_to_x_idx, 0]
        
        max_skill_x_to_y = ccm_skill[x_to_y_idx, 1]
        max_skill_y_to_x = ccm_skill[y_to_x_idx, 2]
        
        # Revised parameters based on paper
        weak_skill_threshold = 0.1
        strong_skill_threshold = 0.3
        lag_threshold = 2
        
        # Calculate additional metrics
        skill_difference = abs(max_skill_x_to_y - max_skill_y_to_x)
        max_skill = max(max_skill_x_to_y, max_skill_y_to_x)
        min_skill = min(max_skill_x_to_y, max_skill_y_to_x)
        
        # Synchrony: Strong unidirectional forcing with opposite lag patterns
        if ((opt_lag_x_to_y * opt_lag_y_to_x) < 0 and  # Opposite lag signs
            max_skill > strong_skill_threshold and      # Strong skill in one direction
            min_skill < weak_skill_threshold):         # Weak skill in other direction
            return "Synchrony", ccm_skill
        
        # Transitive: Progressive weakening of coupling strength and increasing lag
        elif (skill_difference > strong_skill_threshold and           # Clear asymmetry
            max(abs(opt_lag_x_to_y), abs(opt_lag_y_to_x)) > lag_threshold and  # Large lag in at least one direction
            min_skill < weak_skill_threshold and                    # Weak indirect coupling
            max_skill > strong_skill_threshold):                    # Strong direct coupling
            return "Transitive", ccm_skill
        
        # Bidirectional: Both directions show significant skill at negative lags
        elif (opt_lag_x_to_y < 0 and opt_lag_y_to_x < 0 and    # Both negative lags
            min_skill > weak_skill_threshold and              # Both directions show skill
            skill_difference < strong_skill_threshold):       # Similar skill levels
            return "Bidirectional", ccm_skill
            
        return "Unclassified", ccm_skill

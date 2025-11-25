# Python Integration Scripts

## **Python Bridge Server for ML Inference**

### **File 1: `ml_bridge_server.py` - Main Inference Server**

```python
"""
ML Bridge Server for MQL5 Integration
Receives features from MQL5, runs ML inference, returns predictions.

Usage:
    python ml_bridge_server.py --model random_forest --port 9090

Features:
    - Socket-based communication (low latency)
    - Model hot-reloading
    - Request/response logging
    - Multiple model support
    - Graceful error handling
"""

import argparse
import json
import pickle
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# =============================================================================
# Configuration
# =============================================================================

class ServerConfig:
    """Server configuration with sensible defaults."""
    
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 9090
        self.buffer_size = 4096
        self.model_path = Path("models")
        self.log_path = Path("logs")
        self.model_type = "random_forest"  # or "xgboost", "tensorflow", "pytorch"
        self.enable_caching = True
        self.cache_size = 1000
        
        # Create directories
        self.model_path.mkdir(exist_ok=True)
        self.log_path.mkdir(exist_ok=True)


# =============================================================================
# Model Wrapper Classes
# =============================================================================

class BaseModelWrapper:
    """Base class for model wrappers."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.model_version = "unknown"
        self.load_time = None
        
    def load_model(self):
        """Load model from disk."""
        raise NotImplementedError
        
    def predict(self, features: np.ndarray) -> Tuple[float, str, float]:
        """
        Make prediction.
        
        Returns:
            (prediction_score, predicted_class, confidence)
        """
        raise NotImplementedError


class RandomForestWrapper(BaseModelWrapper):
    """Wrapper for sklearn RandomForest models."""
    
    def load_model(self):
        """Load sklearn model."""
        start_time = time.time()
        
        try:
            with open(self.model_path / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
                
            # Load metadata if available
            metadata_path = self.model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names", [])
                    self.model_version = metadata.get("version", "1.0")
            
            self.load_time = time.time() - start_time
            
            logger.info(
                f"Loaded RandomForest model: {self.model_version} "
                f"({self.load_time:.3f}s)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[float, str, float]:
        """Make prediction with RandomForest."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get prediction probabilities
        proba = self.model.predict_proba(features)[0]
        
        # Get class prediction
        predicted_class_idx = np.argmax(proba)
        confidence = proba[predicted_class_idx]
        
        # Convert to BUY/SELL signal
        # Assuming binary classification: 0=SELL, 1=BUY
        if predicted_class_idx == 1:
            prediction_score = confidence
            predicted_class = "BUY"
        else:
            prediction_score = -confidence
            predicted_class = "SELL"
        
        return prediction_score, predicted_class, confidence


class XGBoostWrapper(BaseModelWrapper):
    """Wrapper for XGBoost models."""
    
    def load_model(self):
        """Load XGBoost model."""
        start_time = time.time()
        
        try:
            import xgboost as xgb
            
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path / "model.xgb"))
            
            # Load metadata
            metadata_path = self.model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names", [])
                    self.model_version = metadata.get("version", "1.0")
            
            self.load_time = time.time() - start_time
            
            logger.info(
                f"Loaded XGBoost model: {self.model_version} "
                f"({self.load_time:.3f}s)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[float, str, float]:
        """Make prediction with XGBoost."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        import xgboost as xgb
        
        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
        
        # Get prediction
        prediction = self.model.predict(dmatrix)[0]
        
        # Convert to signal
        if prediction > 0.5:
            prediction_score = prediction
            predicted_class = "BUY"
            confidence = prediction
        else:
            prediction_score = -(1 - prediction)
            predicted_class = "SELL"
            confidence = 1 - prediction
        
        return prediction_score, predicted_class, confidence


class TensorFlowWrapper(BaseModelWrapper):
    """Wrapper for TensorFlow/Keras models."""
    
    def load_model(self):
        """Load TensorFlow model."""
        start_time = time.time()
        
        try:
            import tensorflow as tf
            
            self.model = tf.keras.models.load_model(
                str(self.model_path / "model.h5")
            )
            
            # Load metadata
            metadata_path = self.model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names", [])
                    self.model_version = metadata.get("version", "1.0")
            
            self.load_time = time.time() - start_time
            
            logger.info(
                f"Loaded TensorFlow model: {self.model_version} "
                f"({self.load_time:.3f}s)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[float, str, float]:
        """Make prediction with TensorFlow."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(features, verbose=0)[0]
        
        # Handle different output formats
        if len(prediction) == 1:
            # Single output (binary classification)
            prob = float(prediction[0])
            if prob > 0.5:
                prediction_score = prob
                predicted_class = "BUY"
                confidence = prob
            else:
                prediction_score = -(1 - prob)
                predicted_class = "SELL"
                confidence = 1 - prob
        else:
            # Multi-class output
            predicted_class_idx = np.argmax(prediction)
            confidence = float(prediction[predicted_class_idx])
            
            if predicted_class_idx == 1:
                prediction_score = confidence
                predicted_class = "BUY"
            else:
                prediction_score = -confidence
                predicted_class = "SELL"
        
        return prediction_score, predicted_class, confidence


# =============================================================================
# Prediction Cache
# =============================================================================

class PredictionCache:
    """Simple LRU cache for predictions."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[float, str, float]] = {}
        self.access_order: List[str] = []
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, features: np.ndarray) -> str:
        """Generate cache key from features."""
        # Use hash of first/middle/last features for speed
        if len(features) > 0:
            key_parts = [
                str(features[0]),
                str(features[len(features) // 2]) if len(features) > 1 else "",
                str(features[-1]) if len(features) > 1 else "",
                str(len(features))
            ]
            return "_".join(key_parts)
        return ""
    
    def get(self, features: np.ndarray) -> Optional[Tuple[float, str, float]]:
        """Get cached prediction."""
        key = self._generate_key(features)
        
        if key in self.cache:
            self.hits += 1
            
            # Update access order (move to end)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, features: np.ndarray, prediction: Tuple[float, str, float]):
        """Cache prediction."""
        key = self._generate_key(features)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = prediction
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }


# =============================================================================
# ML Bridge Server
# =============================================================================

class MLBridgeServer:
    """Main server class for ML inference."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model_wrapper: Optional[BaseModelWrapper] = None
        self.cache: Optional[PredictionCache] = None
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        
        # Statistics
        self.requests_received = 0
        self.predictions_made = 0
        self.errors = 0
        self.start_time = time.time()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = self.config.log_path / f"server_{datetime.now():%Y%m%d}.log"
        
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO"
        )
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="1 day",
            retention="7 days"
        )
    
    def initialize(self) -> bool:
        """Initialize server components."""
        logger.info("=" * 70)
        logger.info("ML BRIDGE SERVER INITIALIZATION")
        logger.info("=" * 70)
        
        # Load model
        if not self._load_model():
            return False
        
        # Initialize cache
        if self.config.enable_caching:
            self.cache = PredictionCache(max_size=self.config.cache_size)
            logger.info(f"Prediction cache initialized (size={self.config.cache_size})")
        
        # Create server socket
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.config.host, self.config.port))
            self.server_socket.listen(1)
            
            logger.info(f"Server listening on {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            return False
        
        logger.info("=" * 70)
        logger.info("SERVER READY - Waiting for MQL5 connections...")
        logger.info("=" * 70)
        
        return True
    
    def _load_model(self) -> bool:
        """Load ML model based on configuration."""
        logger.info(f"Loading {self.config.model_type} model...")
        
        try:
            if self.config.model_type == "random_forest":
                self.model_wrapper = RandomForestWrapper(self.config.model_path)
            elif self.config.model_type == "xgboost":
                self.model_wrapper = XGBoostWrapper(self.config.model_path)
            elif self.config.model_type == "tensorflow":
                self.model_wrapper = TensorFlowWrapper(self.config.model_path)
            else:
                logger.error(f"Unknown model type: {self.config.model_type}")
                return False
            
            return self.model_wrapper.load_model()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def start(self):
        """Start the server."""
        self.running = True
        
        try:
            while self.running:
                # Accept connection
                logger.info("Waiting for MQL5 connection...")
                client_socket, address = self.server_socket.accept()
                logger.info(f"✓ Connected: {address}")
                
                # Handle client
                self._handle_client(client_socket)
                
        except KeyboardInterrupt:
            logger.info("\nShutdown signal received...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.stop()
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle messages from connected MQL5 client."""
        buffer = b""
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(self.config.buffer_size)
                
                if not data:
                    logger.warning("Client disconnected")
                    break
                
                buffer += data
                
                # Process complete messages (length-prefixed)
                while len(buffer) >= 4:
                    # Read message length
                    msg_length = struct.unpack("I", buffer[:4])[0]
                    
                    if len(buffer) < 4 + msg_length:
                        break  # Incomplete message
                    
                    # Extract message
                    msg_data = buffer[4:4 + msg_length]
                    buffer = buffer[4 + msg_length:]
                    
                    # Process message
                    response = self._process_message(msg_data)
                    
                    # Send response
                    if response:
                        self._send_response(client_socket, response)
        
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, data: bytes) -> Optional[Dict]:
        """Process incoming message from MQL5."""
        try:
            message = json.loads(data.decode("utf-8"))
            self.requests_received += 1
            
            msg_type = message.get("type")
            
            if msg_type == "predict":
                return self._handle_prediction_request(message)
            
            elif msg_type == "heartbeat":
                return {"type": "heartbeat_response", "status": "ok"}
            
            elif msg_type == "stats":
                return self._get_server_stats()
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                return {"type": "error", "message": "Unknown message type"}
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.errors += 1
            return {"type": "error", "message": str(e)}
    
    def _handle_prediction_request(self, message: Dict) -> Dict:
        """Handle prediction request from MQL5."""
        try:
            # Extract features
            features = np.array(message.get("features", []), dtype=np.float64)
            
            if len(features) == 0:
                return {
                    "type": "error",
                    "message": "No features provided"
                }
            
            # Check cache
            cached_result = None
            if self.cache:
                cached_result = self.cache.get(features)
            
            if cached_result:
                prediction_score, predicted_class, confidence = cached_result
                logger.debug(f"Cache HIT: {predicted_class} (score={prediction_score:.4f})")
            else:
                # Make prediction
                start_time = time.time()
                
                prediction_score, predicted_class, confidence = \
                    self.model_wrapper.predict(features)
                
                inference_time = (time.time() - start_time) * 1_000_000  # microseconds
                
                # Cache result
                if self.cache:
                    self.cache.set(features, (prediction_score, predicted_class, confidence))
                
                self.predictions_made += 1
                
                logger.info(
                    f"Prediction: {predicted_class} "
                    f"(score={prediction_score:.4f}, conf={confidence:.4f}, "
                    f"latency={inference_time:.0f}µs)"
                )
            
            # Build response
            response = {
                "type": "prediction",
                "prediction_score": float(prediction_score),
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "model_version": self.model_wrapper.model_version,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            self.errors += 1
            return {
                "type": "error",
                "message": str(e)
            }
    
    def _get_server_stats(self) -> Dict:
        """Get server statistics."""
        uptime = time.time() - self.start_time
        
        stats = {
            "type": "stats",
            "uptime_seconds": uptime,
            "requests_received": self.requests_received,
            "predictions_made": self.predictions_made,
            "errors": self.errors,
            "model_version": self.model_wrapper.model_version if self.model_wrapper else "unknown"
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    def _send_response(self, client_socket: socket.socket, response: Dict):
        """Send response to MQL5 client."""
        try:
            # Serialize response
            data = json.dumps(response).encode("utf-8")
            
            # Send with length prefix
            length = struct.pack("I", len(data))
            client_socket.sendall(length + data)
            
        except Exception as e:
            logger.error(f"Error sending response: {e}")
    
    def stop(self):
        """Stop the server."""
        self.running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        # Print final statistics
        logger.info("=" * 70)
        logger.info("SERVER SHUTDOWN")
        logger.info("=" * 70)
        
        stats = self._get_server_stats()
        logger.info(f"Total requests: {stats['requests_received']}")
        logger.info(f"Total predictions: {stats['predictions_made']}")
        logger.info(f"Total errors: {stats['errors']}")
        logger.info(f"Uptime: {stats['uptime_seconds']:.0f}s")
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.2f}%")
        
        logger.info("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML Bridge Server for MQL5 Integration"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "tensorflow"],
        help="Model type to load"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Server port"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="models",
        help="Path to model files"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prediction caching"
    )
    
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1000,
        help="Cache size (number of entries)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ServerConfig()
    config.model_type = args.model
    config.port = args.port
    config.model_path = Path(args.model_path)
    config.enable_caching = not args.no_cache
    config.cache_size = args.cache_size
    
    # Create and start server
    server = MLBridgeServer(config)
    
    if not server.initialize():
        logger.error("Failed to initialize server")
        sys.exit(1)
    
    server.start()


if __name__ == "__main__":
    main()
```

---

### **File 2: `log_analyzer.py` - Analyze MQL5 Logs**

```python
"""
MQL5 Log Analyzer
Analyzes structured logs generated by CMLLogger.

Usage:
    python log_analyzer.py --log MLLogs/MLPoweredEA/20241118.csv
    
Features:
    - Parse structured CSV logs
    - Generate performance reports
    - Detect anomalies
    - Visualize metrics
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


class MQL5LogAnalyzer:
    """Analyzer for MQL5 structured logs."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.df: Optional[pd.DataFrame] = None
        self.model_logs: Optional[pd.DataFrame] = None
        self.trade_logs: Optional[pd.DataFrame] = None
        
    def load_logs(self) -> bool:
        """Load and parse log file."""
        try:
            self.df = pd.read_csv(self.log_file)
            
            # Parse timestamps
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            
            # Separate by log type
            if 'Level' in self.df.columns:
                # General logs
                logger.info(f"Loaded {len(self.df)} log entries")
                
                # Extract model inference logs
                model_mask = self.df['Level'] == 'MODEL_INFERENCE'
                if model_mask.any():
                    self.model_logs = self.df[model_mask].copy()
                    logger.info(f"Found {len(self.model_logs)} model inference logs")
                
                # Extract trade execution logs
                trade_mask = self.df['Level'] == 'TRADE_EXECUTION'
                if trade_mask.any():
                    self.trade_logs = self.df[trade_mask].copy()
                    logger.info(f"Found {len(self.trade_logs)} trade execution logs")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load logs: {e}")
            return False
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        report = {
            "summary": self._generate_summary(),
            "model_performance": self._analyze_model_performance(),
            "trade_performance": self._analyze_trade_performance(),
            "anomalies": self._detect_anomalies()
        }
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        if self.df is None:
            return {}
        
        summary = {
            "total_entries": len(self.df),
            "time_range": (
                self.df['Timestamp'].min().strftime("%Y-%m-%d %H:%M:%S"),
                self.df['Timestamp'].max().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "duration_hours": (
                self.df['Timestamp'].max() - self.df['Timestamp'].min()
            ).total_seconds() / 3600
        }
        
        # Count by level
        if 'Level' in self.df.columns:
            level_counts = self.df['Level'].value_counts().to_dict()
            summary["entries_by_level"] = level_counts
        
        return summary
    
    def _analyze_model_performance(self) -> Dict:
        """Analyze model inference performance."""
        if self.model_logs is None or len(self.model_logs) == 0:
            return {}
        
        # Parse model-specific columns
        # Assuming CSV format: Timestamp,Strategy,MagicNumber,Level,PredictionScore,PredictedClass,LatencyUS,ModelVersion
        
        analysis = {}
        
        # Latency analysis
        if 'LatencyUS' in self.model_logs.columns:
            latencies = pd.to_numeric(self.model_logs['LatencyUS'], errors='coerce')
            latencies = latencies.dropna()
            
            analysis["latency"] = {
                "mean_us": float(latencies.mean()),
                "median_us": float(latencies.median()),
                "p95_us": float(latencies.quantile(0.95)),
                "p99_us": float(latencies.quantile(0.99)),
                "max_us": float(latencies.max())
            }
        
        # Prediction distribution
        if 'PredictionScore' in self.model_logs.columns:
            scores = pd.to_numeric(self.model_logs['PredictionScore'], errors='coerce')
            scores = scores.dropna()
            
            analysis["prediction_distribution"] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "quartiles": {
                    "q25": float(scores.quantile(0.25)),
                    "q50": float(scores.quantile(0.50)),
                    "q75": float(scores.quantile(0.75))
                }
            }
        
        # Class distribution
        if 'PredictedClass' in self.model_logs.columns:
            class_counts = self.model_logs['PredictedClass'].value_counts().to_dict()
            total = sum(class_counts.values())
            class_pct = {k: v/total*100 for k, v in class_counts.items()}
            
            analysis["class_distribution"] = {
                "counts": class_counts,
                "percentages": class_pct
            }
        
        return analysis
    
    def _analyze_trade_performance(self) -> Dict:
        """Analyze trade execution performance."""
        if self.trade_logs is None or len(self.trade_logs) == 0:
            return {}
        
        # Parse trade-specific columns
        # Format: Timestamp,Strategy,MagicNumber,Level,Ticket,SignalType,EntryPrice,SL,TP,PositionSize,Confidence
        
        analysis = {}
        
        # Signal distribution
        if 'SignalType' in self.trade_logs.columns:
            signal_counts = self.trade_logs['SignalType'].value_counts().to_dict()
            total = sum(signal_counts.values())
            signal_pct = {k: v/total*100 for k, v in signal_counts.items()}
            
            analysis["signal_distribution"] = {
                "counts": signal_counts,
                "percentages": signal_pct
            }
        
        # Confidence analysis
        if 'Confidence' in self.trade_logs.columns:
            confidence = pd.to_numeric(self.trade_logs['Confidence'], errors='coerce')
            confidence = confidence.dropna()
            
            analysis["confidence"] = {
                "mean": float(confidence.mean()),
                "median": float(confidence.median()),
                "min": float(confidence.min()),
                "max": float(confidence.max())
            }
        
        # Position sizing
        if 'PositionSize' in self.trade_logs.columns:
            sizes = pd.to_numeric(self.trade_logs['PositionSize'], errors='coerce')
            sizes = sizes.dropna()
            
            analysis["position_size"] = {
                "mean": float(sizes.mean()),
                "median": float(sizes.median()),
                "min": float(sizes.min()),
                "max": float(sizes.max())
            }
        
        return analysis
    
    def _detect_anomalies(self) -> Dict:
        """Detect anomalies in logs."""
        anomalies = {
            "high_latency_events": [],
            "extreme_predictions": [],
            "errors": []
        }

        # High latency events (> 1 second = 1,000,000 microseconds)
        if self.model_logs is not None and 'LatencyUS' in self.model_logs.columns:
            latencies = pd.to_numeric(self.model_logs['LatencyUS'], errors='coerce')
            high_latency_mask = latencies > 1_000_000
            
            if high_latency_mask.any():
                high_latency_events = self.model_logs[high_latency_mask][
                    ['Timestamp', 'LatencyUS']
                ].to_dict('records')
                
                anomalies["high_latency_events"] = high_latency_events
                logger.warning(f"Found {len(high_latency_events)} high latency events (>1s)")
        
        # Extreme predictions (|score| > 0.95)
        if self.model_logs is not None and 'PredictionScore' in self.model_logs.columns:
            scores = pd.to_numeric(self.model_logs['PredictionScore'], errors='coerce')
            extreme_mask = scores.abs() > 0.95
            
            if extreme_mask.any():
                extreme_predictions = self.model_logs[extreme_mask][
                    ['Timestamp', 'PredictionScore', 'PredictedClass']
                ].to_dict('records')
                
                anomalies["extreme_predictions"] = extreme_predictions
                logger.info(f"Found {len(extreme_predictions)} extreme predictions (|score|>0.95)")
        
        # Error entries
        if self.df is not None and 'Level' in self.df.columns:
            error_mask = self.df['Level'].isin(['ERROR', 'FATAL'])
            
            if error_mask.any():
                errors = self.df[error_mask][
                    ['Timestamp', 'Level', 'Message', 'Function']
                ].to_dict('records')
                
                anomalies["errors"] = errors
                logger.warning(f"Found {len(errors)} error entries")
        
        return anomalies
    
    def plot_latency_over_time(self, save_path: Optional[Path] = None):
        """Plot inference latency over time."""
        if self.model_logs is None or 'LatencyUS' not in self.model_logs.columns:
            logger.warning("No latency data to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Convert to milliseconds for readability
        latencies_ms = pd.to_numeric(self.model_logs['LatencyUS'], errors='coerce') / 1000
        timestamps = self.model_logs['Timestamp']
        
        plt.plot(timestamps, latencies_ms, alpha=0.6, linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Latency (ms)')
        plt.title('Model Inference Latency Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add mean line
        mean_latency = latencies_ms.mean()
        plt.axhline(y=mean_latency, color='r', linestyle='--', 
                   label=f'Mean: {mean_latency:.2f}ms')
        
        # Add p95 line
        p95_latency = latencies_ms.quantile(0.95)
        plt.axhline(y=p95_latency, color='orange', linestyle='--', 
                   label=f'P95: {p95_latency:.2f}ms')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved latency plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_prediction_distribution(self, save_path: Optional[Path] = None):
        """Plot distribution of prediction scores."""
        if self.model_logs is None or 'PredictionScore' not in self.model_logs.columns:
            logger.warning("No prediction data to plot")
            return
        
        scores = pd.to_numeric(self.model_logs['PredictionScore'], errors='coerce')
        scores = scores.dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='r', linestyle='--', label='Neutral')
        axes[0].set_xlabel('Prediction Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Prediction Scores')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(scores, vert=True)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Prediction Score')
        axes[1].set_title('Prediction Score Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved prediction distribution plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_distribution(self, save_path: Optional[Path] = None):
        """Plot distribution of predicted classes."""
        if self.model_logs is None or 'PredictedClass' not in self.model_logs.columns:
            logger.warning("No class data to plot")
            return
        
        class_counts = self.model_logs['PredictedClass'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        class_counts.plot(kind='bar', ax=axes[0], alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Predicted Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Predicted Class Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        axes[1].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', 
                   startangle=90)
        axes[1].set_title('Predicted Class Proportions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved class distribution plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_latency_heatmap(self, save_path: Optional[Path] = None):
        """Plot latency heatmap by hour of day and day of week."""
        if self.model_logs is None or 'LatencyUS' not in self.model_logs.columns:
            logger.warning("No latency data to plot")
            return
        
        df = self.model_logs.copy()
        df['LatencyMS'] = pd.to_numeric(df['LatencyUS'], errors='coerce') / 1000
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        
        # Create pivot table
        pivot = df.pivot_table(
            values='LatencyMS',
            index='DayOfWeek',
            columns='Hour',
            aggfunc='mean'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([day for day in day_order if day in pivot.index])
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Latency (ms)'})
        plt.title('Average Inference Latency by Day and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved latency heatmap to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_summary_report(self, output_path: Path):
        """Export comprehensive summary report as HTML."""
        report = self.generate_performance_report()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MQL5 Log Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .metric {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .metric-label {{
                    font-weight: bold;
                    color: #7f8c8d;
                }}
                .metric-value {{
                    font-size: 1.2em;
                    color: #2c3e50;
                }}
                .warning {{
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .error {{
                    background-color: #f8d7da;
                    border-left: 4px solid #dc3545;
                    padding: 15px;
                    margin: 10px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MQL5 Log Analysis Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Log File:</strong> {self.log_file}</p>
        """
        
        # Summary section
        if 'summary' in report:
            summary = report['summary']
            html += f"""
                <h2>Summary</h2>
                <div class="metric">
                    <span class="metric-label">Total Entries:</span>
                    <span class="metric-value">{summary.get('total_entries', 0):,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Time Range:</span>
                    <span class="metric-value">{summary.get('time_range', ['N/A', 'N/A'])[0]} to {summary.get('time_range', ['N/A', 'N/A'])[1]}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Duration:</span>
                    <span class="metric-value">{summary.get('duration_hours', 0):.2f} hours</span>
                </div>
            """
        
        # Model performance section
        if 'model_performance' in report and report['model_performance']:
            model_perf = report['model_performance']
            html += "<h2>Model Performance</h2>"
            
            if 'latency' in model_perf:
                latency = model_perf['latency']
                html += f"""
                <h3>Inference Latency</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value (ms)</th>
                    </tr>
                    <tr>
                        <td>Mean</td>
                        <td>{latency['mean_us']/1000:.3f}</td>
                    </tr>
                    <tr>
                        <td>Median</td>
                        <td>{latency['median_us']/1000:.3f}</td>
                    </tr>
                    <tr>
                        <td>P95</td>
                        <td>{latency['p95_us']/1000:.3f}</td>
                    </tr>
                    <tr>
                        <td>P99</td>
                        <td>{latency['p99_us']/1000:.3f}</td>
                    </tr>
                    <tr>
                        <td>Max</td>
                        <td>{latency['max_us']/1000:.3f}</td>
                    </tr>
                </table>
                """
            
            if 'prediction_distribution' in model_perf:
                pred_dist = model_perf['prediction_distribution']
                html += f"""
                <h3>Prediction Distribution</h3>
                <table>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Mean</td>
                        <td>{pred_dist['mean']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Std Dev</td>
                        <td>{pred_dist['std']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Min</td>
                        <td>{pred_dist['min']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Max</td>
                        <td>{pred_dist['max']:.4f}</td>
                    </tr>
                </table>
                """
            
            if 'class_distribution' in model_perf:
                class_dist = model_perf['class_distribution']
                html += f"""
                <h3>Class Distribution</h3>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                """
                for cls, count in class_dist['counts'].items():
                    pct = class_dist['percentages'][cls]
                    html += f"""
                    <tr>
                        <td>{cls}</td>
                        <td>{count}</td>
                        <td>{pct:.2f}%</td>
                    </tr>
                    """
                html += "</table>"
        
        # Anomalies section
        if 'anomalies' in report:
            anomalies = report['anomalies']
            html += "<h2>Anomalies</h2>"
            
            if anomalies.get('high_latency_events'):
                html += f"""
                <div class="warning">
                    <strong>⚠️ High Latency Events:</strong> 
                    Found {len(anomalies['high_latency_events'])} events with latency > 1 second
                </div>
                """
            
            if anomalies.get('extreme_predictions'):
                html += f"""
                <div class="warning">
                    <strong>⚠️ Extreme Predictions:</strong> 
                    Found {len(anomalies['extreme_predictions'])} predictions with |score| > 0.95
                </div>
                """
            
            if anomalies.get('errors'):
                html += f"""
                <div class="error">
                    <strong>❌ Errors:</strong> 
                    Found {len(anomalies['errors'])} error entries
                </div>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Level</th>
                        <th>Message</th>
                    </tr>
                """
                for error in anomalies['errors'][:10]:  # Show first 10
                    html += f"""
                    <tr>
                        <td>{error.get('Timestamp', 'N/A')}</td>
                        <td>{error.get('Level', 'N/A')}</td>
                        <td>{error.get('Message', 'N/A')}</td>
                    </tr>
                    """
                html += "</table>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Exported summary report to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze MQL5 structured logs"
    )
    
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to log file (CSV format)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Output directory for reports and plots"
    )
    
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer
    analyzer = MQL5LogAnalyzer(Path(args.log))
    
    # Load logs
    if not analyzer.load_logs():
        logger.error("Failed to load logs")
        return
    
    # Generate report
    logger.info("Generating performance report...")
    report = analyzer.generate_performance_report()
    
    # Print summary to console
    print("\n" + "="*70)
    print("MQL5 LOG ANALYSIS SUMMARY")
    print("="*70)
    
    if 'summary' in report:
        summary = report['summary']
        print(f"\nTotal Entries: {summary.get('total_entries', 0):,}")
        print(f"Time Range: {summary.get('time_range', ['N/A', 'N/A'])[0]} to {summary.get('time_range', ['N/A', 'N/A'])[1]}")
        print(f"Duration: {summary.get('duration_hours', 0):.2f} hours")
    
    if 'model_performance' in report and report['model_performance']:
        model_perf = report['model_performance']
        print("\nModel Performance:")
        
        if 'latency' in model_perf:
            latency = model_perf['latency']
            print(f"  Avg Latency: {latency['mean_us']/1000:.2f}ms")
            print(f"  P95 Latency: {latency['p95_us']/1000:.2f}ms")
        
        if 'class_distribution' in model_perf:
            class_dist = model_perf['class_distribution']
            print("\n  Class Distribution:")
            for cls, pct in class_dist['percentages'].items():
                print(f"    {cls}: {pct:.2f}%")
    
    if 'anomalies' in report:
        anomalies = report['anomalies']
        if any(anomalies.values()):
            print("\nAnomalies Detected:")
            if anomalies.get('high_latency_events'):
                print(f"  ⚠️  High latency events: {len(anomalies['high_latency_events'])}")
            if anomalies.get('extreme_predictions'):
                print(f"  ⚠️  Extreme predictions: {len(anomalies['extreme_predictions'])}")
            if anomalies.get('errors'):
                print(f"  ❌ Errors: {len(anomalies['errors'])}")
    
    print("="*70 + "\n")
    
    # Export HTML report
    html_path = output_dir / "analysis_report.html"
    analyzer.export_summary_report(html_path)
    print(f"✓ Exported HTML report: {html_path}")
    
    # Generate plots if requested
    if args.plots:
        logger.info("Generating visualization plots...")
        
        analyzer.plot_latency_over_time(output_dir / "latency_over_time.png")
        analyzer.plot_prediction_distribution(output_dir / "prediction_distribution.png")
        analyzer.plot_class_distribution(output_dir / "class_distribution.png")
        analyzer.plot_latency_heatmap(output_dir / "latency_heatmap.png")
        
        print(f"✓ Generated plots in: {output_dir}")


if __name__ == "__main__":
    main()
```

---

### **File 3: `cache_monitor.py` - Real-time Cache Monitoring**

```python
"""
Real-time Cache Monitoring Dashboard
Monitors MQL5 cache performance in real-time.

Usage:
    python cache_monitor.py --cache-stats MLCache/cache_stats.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


class CacheMonitor:
    """Real-time cache performance monitor."""
    
    def __init__(self, stats_file: Path, history_size: int = 100):
        self.stats_file = stats_file
        self.history_size = history_size
        
        # Data buffers
        self.timestamps = deque(maxlen=history_size)
        self.hit_rates = deque(maxlen=history_size)
        self.total_hits = deque(maxlen=history_size)
        self.total_misses = deque(maxlen=history_size)
        
        # Previous state for delta calculation
        self.prev_hits = 0
        self.prev_misses = 0
    
    def read_stats(self) -> Optional[Dict]:
        """Read current cache statistics."""
        try:
            if not self.stats_file.exists():
                return None
            
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            
            return stats
            
        except Exception as e:
            print(f"Error reading stats: {e}")
            return None
    
    def update_data(self, stats: Dict):
        """Update monitoring data."""
        current_time = datetime.now()
        
        # Extract metrics
        hits = stats.get('total_hits', 0)
        misses = stats.get('total_misses', 0)
        total = hits + misses
        
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        # Update buffers
        self.timestamps.append(current_time)
        self.hit_rates.append(hit_rate)
        self.total_hits.append(hits)
        self.total_misses.append(misses)
        
        # Update previous state
        self.prev_hits = hits
        self.prev_misses = misses
    
    def start_monitoring(self, update_interval: int = 1):
        """Start real-time monitoring with live plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('MQL5 Cache Performance Monitor', fontsize=16)
        
        def animate(frame):
            # Read latest stats
            stats = self.read_stats()
            
            if stats:
                self.update_data(stats)
            
            # Clear all axes
            for ax in axes.flat:
                ax.clear()
            
            if len(self.timestamps) == 0:
                return
            
            # Plot 1: Hit Rate Over Time
            axes[0, 0].plot(self.timestamps, self.hit_rates, 'b-', linewidth=2)
            axes[0, 0].set_title('Cache Hit Rate')
            axes[0, 0].set_ylabel('Hit Rate (%)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim([0, 100])
            
            # Plot 2: Hits vs Misses
            axes[0, 1].plot(self.timestamps, self.total_hits, 'g-', label='Hits', linewidth=2)
            axes[0, 1].plot(self.timestamps, self.total_misses, 'r-', label='Misses', linewidth=2)
            axes[0, 1].set_title('Cumulative Hits vs Misses')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Current Statistics (text)
            axes[1, 0].axis('off')
            if stats:
                stats_text = f"""
                Current Statistics:
                
                Total Hits: {stats.get('total_hits', 0):,}
                Total Misses: {stats.get('total_misses', 0):,}
                Hit Rate: {self.hit_rates[-1] if self.hit_rates else 0:.2f}%
                
                Cache Size: {stats.get('cache_size', 0)} / {stats.get('max_size', 0)}
                
                Last Update: {datetime.now().strftime('%H:%M:%S')}
                """
                axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, 
                              verticalalignment='center', family='monospace')
            
            # Plot 4: Hit Rate Distribution
            if len(self.hit_rates) > 10:
                axes[1, 1].hist(self.hit_rates, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Hit Rate Distribution')
                axes[1, 1].set_xlabel('Hit Rate (%)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
        
        # Start animation
        ani = animation.FuncAnimation(
            fig, animate, interval=update_interval*1000, cache_frame_data=False
        )
        
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time MQL5 cache monitoring"
    )
    
    parser.add_argument(
        "--cache-stats",
        type=str,
        required=True,
        help="Path to cache statistics file (JSON)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Update interval in seconds"
    )
    
    parser.add_argument(
        "--history",
        type=int,
        default=100,
        help="Number of data points to keep in history"
    )
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = CacheMonitor(
        Path(args.cache_stats),
        history_size=args.history
    )
    
    # Start monitoring
    print(f"Starting cache monitor (updating every {args.interval}s)...")
    print(f"Reading from: {args.cache_stats}")
    print("Press Ctrl+C to stop")
    
    try:
        monitor.start_monitoring(update_interval=args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    main()
```

---

### **File 4: `train_example_model.py` - Training Script**

```python
"""
Example: Train a model for MQL5 integration
Demonstrates how to prepare a model for the ML Bridge Server.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def generate_synthetic_features(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic trading features for demonstration."""
    np.random.seed(42)
    
    # Feature 1-3: Moving averages (simulated)
    ma_10 = np.random.randn(n_samples) * 0.01 + 1.1000
    ma_20 = np.random.randn(n_samples) * 0.01 + 1.1005
    ma_50 = np.random.randn(n_samples) * 0.01 + 1.1010
    
    # Feature 4-5: RSI variations
    rsi_14 = np.random.uniform(20, 80, n_samples)
    rsi_28 = np.random.uniform(20, 80, n_samples)
    
    # Feature 6-7: Volatility measures
    atr = np.random.uniform(0.0010, 0.0050, n_samples)
    std_dev = np.random.uniform(0.0005, 0.0030, n_samples)
    
    # Feature 8-9: Volume indicators
    vol_ma = np.random.uniform(1000, 5000, n_samples)
    vol_ratio = np.random.uniform(0.5, 2.0, n_samples)
    
    # Feature 10: Momentum
    momentum = np.random.randn(n_samples) * 0.005
    
    df = pd.DataFrame({
        'ma_10': ma_10,
        'ma_20': ma_20,
        'ma_50': ma_50,
        'rsi_14': rsi_14,
        'rsi_28': rsi_28,
        'atr': atr,
        'std_dev': std_dev,
        'vol_ma': vol_ma,
        'vol_ratio': vol_ratio,
        'momentum': momentum
    })
    
    return df


def generate_labels(features: pd.DataFrame) -> np.ndarray:
    """Generate synthetic labels based on features."""
    # Simple rule: BUY (1) if momentum > 0 and RSI < 70, else SELL (0)
    labels = np.where(
        (features['momentum'] > 0) & (features['rsi_14'] < 70),
        1,  # BUY
        0   # SELL
    )
    
    return labels


def train_and_save_model():
    """Train a RandomForest model and save for ML Bridge Server."""
    print("=" * 70)
    print("TRAINING ML MODEL FOR MQL5 INTEGRATION")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic training data...")
    features = generate_synthetic_features(n_samples=10000)
    labels = generate_labels(features)
    
    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Class distribution: BUY={np.sum(labels)}, SELL={len(labels)-np.sum(labels)}")
    
    # Split data
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train model
    print("\n3. Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Training complete")
    
    # Evaluate
    print("\n4. Evaluating model...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"   Train accuracy: {train_score:.4f}")
    print(f"   Test accuracy: {test_score:.4f}")
    
    y_pred = model.predict(X_test)
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['SELL', 'BUY']))
    
    print("\n   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\n5. Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']:15s}: {row['importance']:.4f}")
    
    # Save model
    print("\n6. Saving model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Save model
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved to {model_dir / 'model.pkl'}")
    
    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "version": "1.0",
        "feature_names": features.columns.tolist(),
        "n_features": len(features.columns),
        "train_accuracy": float(train_score),
        "test_accuracy": float(test_score),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "created": pd.Timestamp.now().isoformat()
    }
    
    import json
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, indent=2, fp=f)
    print(f"   ✓ Metadata saved to {model_dir / 'metadata.json'}")
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start ML Bridge Server:")
    print("   python ml_bridge_server.py --model random_forest --port 9090")
    print("\n2. Configure MQL5 EA to connect to localhost:9090")
    print("\n3. Monitor logs and cache performance")
    print("=" * 70)


if __name__ == "__main__":
    train_and_save_model()
```

---

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# Global benchmark instance
benchmark_endpoints = {}


def load_xgboost_benchmark():
    # Install XGBoost and required dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pyarrow"]
    )  # For parquet support
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "fastparquet"]
    )  # Alternative parquet engine
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openml==0.12.2"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/opt/HPOBench"]
    )

    # Import and initialize XGBoost benchmark (new version)
    from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark  # type: ignore

    benchmark = XGBoostBenchmark(task_id=43)  # heart-c with string label fix

    # Define benchmark-specific handlers
    def handle_objective_function(handler, data):
        config_dict = data["config"]
        fidelity = data.get("fidelity", {"n_estimators": 100, "subsample": 0.8})
        rng = data.get("rng", 1)

        # Convert dict to ConfigSpace configuration
        config_space = benchmark.get_configuration_space(seed=1)
        from ConfigSpace import Configuration  # type: ignore

        config = Configuration(config_space, config_dict)

        result = benchmark.objective_function(configuration=config, rng=rng)
        return result

    def handle_sample_config(handler, data):
        config_space = benchmark.get_configuration_space(seed=1)
        config = config_space.sample_configuration()
        return dict(config)

    # Register benchmark endpoints
    benchmark_endpoints.update(
        {
            "/objective_function": handle_objective_function,
            "/sample_config": handle_sample_config,
        }
    )

    return {
        "status": "xgboost benchmark loaded successfully",
        "endpoints": list(benchmark_endpoints.keys()),
    }


def load_nn_benchmark():
    # Install HPOBench with NN support
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    # Install compatible version of OpenML that has set_cache_directory
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openml==0.12.2"])

    # Import and initialize NN benchmark
    from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark  # type: ignore

    benchmark = NNBenchmark(task_id=167149)

    def handle_objective_function(handler, data):
        config_dict = data["config"]
        # fidelity = data.get("fidelity", {"iter": 128, "subsample": 0.5})
        rng = data.get("rng", 1)

        # Convert dict to ConfigSpace configuration
        config_space = benchmark.get_configuration_space(seed=1)
        from ConfigSpace import Configuration  # type: ignore

        config = Configuration(config_space, config_dict)

        result = benchmark.objective_function(configuration=config, rng=rng)
        return result

    def handle_sample_config(handler, data):
        config_space = benchmark.get_configuration_space(seed=1)
        config = config_space.sample_configuration()
        return dict(config)

    # Register benchmark endpoints
    benchmark_endpoints.update(
        {
            "/objective_function": handle_objective_function,
            "/sample_config": handle_sample_config,
        }
    )

    return {
        "status": "nn benchmark loaded successfully",
        "endpoints": list(benchmark_endpoints.keys()),
    }


def load_svm_benchmark():
    # Install HPOBench with SVM support
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/opt/HPOBench[svm]"]
    )
    # Install parquet support with compatible pandas version
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pandas>=1.0.0"]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])

    # Import and initialize SVM benchmark
    from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark  # type: ignore

    benchmark = SVMBenchmark(task_id=167149)

    def handle_objective_function(handler, data):
        config_dict = data["config"]
        fidelity = data.get("fidelity", {"subsample": 0.5})
        rng = data.get("rng", 1)

        # Convert dict to ConfigSpace configuration

        config_space = benchmark.get_configuration_space(seed=1)
        from ConfigSpace import Configuration  # type: ignore

        config = Configuration(config_space, config_dict)

        result = benchmark.objective_function(
            configuration=config, fidelity=fidelity, rng=rng
        )
        return result

    def handle_sample_config(handler, data):
        config_space = benchmark.get_configuration_space(seed=1)
        config = config_space.sample_configuration()
        return dict(config)

    # Register benchmark endpoints
    benchmark_endpoints.update(
        {
            "/objective_function": handle_objective_function,
            "/sample_config": handle_sample_config,
        }
    )

    return {
        "status": "svm benchmark loaded successfully",
        "endpoints": list(benchmark_endpoints.keys()),
    }


def load_pybnn_benchmark():
    # Disable SSL verification for certificate issues
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/opt/HPOBench[pybnn]"]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openml==0.12.2"])

    from hpobench.benchmarks.ml.pybnn import BNNOnBostonHousing  # type: ignore

    benchmark = BNNOnBostonHousing()

    def handle_objective_function(handler, data):
        config_dict = data["config"]
        rng = data.get("rng", 1)

        config_space = benchmark.get_configuration_space(seed=1)
        from ConfigSpace import Configuration  # type: ignore

        config = Configuration(config_space, config_dict)

        result = benchmark.objective_function(configuration=config, rng=rng)
        return result

    def handle_sample_config(handler, data):
        # import numpy as np

        config_space = benchmark.get_configuration_space(seed=1)
        # Set the random state directly on the configuration space for varied sampling
        config_space.seed(1)
        config = config_space.sample_configuration()
        return dict(config)

    benchmark_endpoints.update(
        {
            "/objective_function": handle_objective_function,
            "/sample_config": handle_sample_config,
        }
    )

    return {
        "status": "pybnn benchmark loaded successfully",
        "endpoints": list(benchmark_endpoints.keys()),
    }


def load_histgb_benchmark():
    # Install HPOBench with HistGB support and dependencies
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "scikit-learn>=0.24.0"]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openml==0.12.2"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/opt/HPOBench"]
    )

    from hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark  # type: ignore

    # Initialize with task_id like other benchmarks
    benchmark = HistGBBenchmark(task_id=167149)

    def handle_objective_function(handler, data):
        config_dict = data["config"]
        # Use proper fidelity parameter names based on histgb_benchmark.py
        fidelity = data.get("fidelity", {"n_estimators": 100, "subsample": 0.8})
        rng = data.get("rng", 1)

        # Convert dict to ConfigSpace configuration
        config_space = benchmark.get_configuration_space(seed=1)
        from ConfigSpace import Configuration  # type: ignore

        config = Configuration(config_space, config_dict)

        result = benchmark.objective_function(
            configuration=config, fidelity=fidelity, rng=rng
        )
        return result

    def handle_sample_config(handler, data):
        config_space = benchmark.get_configuration_space(seed=1)
        config = config_space.sample_configuration()
        return dict(config)

    # Register benchmark endpoints
    benchmark_endpoints.update(
        {
            "/objective_function": handle_objective_function,
            "/sample_config": handle_sample_config,
        }
    )

    return {
        "status": "histgb benchmark loaded successfully",
        "endpoints": list(benchmark_endpoints.keys()),
    }


def load_lr_benchmark():
    # Install HPOBench with Logistic Regression support and dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openml==0.12.2"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/opt/HPOBench"]
    )

    from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark  # type: ignore

    # Initialize with task_id like other benchmarks
    benchmark = LRBenchmark(task_id=167149)

    def handle_objective_function(handler, data):
        config_dict = data["config"]
        # Use proper fidelity parameter names based on lr_benchmark.py
        fidelity = data.get("fidelity", {"iter": 1000, "subsample": 0.8})
        rng = data.get("rng", 1)

        # Convert dict to ConfigSpace configuration
        config_space = benchmark.get_configuration_space(seed=1)
        from ConfigSpace import Configuration  # type: ignore

        config = Configuration(config_space, config_dict)

        result = benchmark.objective_function(
            configuration=config, fidelity=fidelity, rng=rng
        )
        return result

    def handle_sample_config(handler, data):
        config_space = benchmark.get_configuration_space(seed=1)
        config = config_space.sample_configuration()
        return dict(config)

    # Register benchmark endpoints
    benchmark_endpoints.update(
        {
            "/objective_function": handle_objective_function,
            "/sample_config": handle_sample_config,
        }
    )

    return {
        "status": "lr benchmark loaded successfully",
        "endpoints": list(benchmark_endpoints.keys()),
    }


def load_rf_benchmark():
    # Install HPOBench with Random Forest support and dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openml==0.12.2"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/opt/HPOBench"]
    )

    from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark  # type: ignore

    # Initialize with task_id like other benchmarks
    benchmark = RandomForestBenchmark(task_id=167149)

    def handle_objective_function(handler, data):
        config_dict = data["config"]
        # Use proper fidelity parameter names based on rf_benchmark.py
        fidelity = data.get("fidelity", {"n_estimators": 100, "subsample": 0.8})
        rng = data.get("rng", 1)

        # Convert dict to ConfigSpace configuration
        config_space = benchmark.get_configuration_space(seed=1)
        from ConfigSpace import Configuration  # type: ignore

        config = Configuration(config_space, config_dict)

        result = benchmark.objective_function(
            configuration=config, fidelity=fidelity, rng=rng
        )
        return result

    def handle_sample_config(handler, data):
        config_space = benchmark.get_configuration_space(seed=1)
        config = config_space.sample_configuration()
        return dict(config)

    # Register benchmark endpoints
    benchmark_endpoints.update(
        {
            "/objective_function": handle_objective_function,
            "/sample_config": handle_sample_config,
        }
    )

    return {
        "status": "rf benchmark loaded successfully",
        "endpoints": list(benchmark_endpoints.keys()),
    }


# Dictionary mapping benchmark names to loader functions
BENCHMARK_LOADERS = {
    "xgboost": load_xgboost_benchmark,
    "svm": load_svm_benchmark,
    "nn": load_nn_benchmark,
    "pybnn": load_pybnn_benchmark,
    "histgb": load_histgb_benchmark,
    "lr": load_lr_benchmark,
    "rf": load_rf_benchmark,
}


def load_benchmark(handler, data):
    benchmark_name = data.get("benchmark")

    if not benchmark_name:
        return {"error": "benchmark parameter is required"}

    if benchmark_name not in BENCHMARK_LOADERS:
        return {
            "error": f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARK_LOADERS.keys())}"
        }

    try:
        return BENCHMARK_LOADERS[benchmark_name]()
    except Exception as e:
        return {"error": f"Failed to load {benchmark_name}: {str(e)}"}


# Base endpoint routing
BASE_ENDPOINTS = {
    "/load": load_benchmark,
}


class HPOHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Check base endpoints first
        if self.path in BASE_ENDPOINTS:
            self.handle_request(BASE_ENDPOINTS[self.path])
        # Then check benchmark-specific endpoints
        elif self.path in benchmark_endpoints:
            self.handle_request(benchmark_endpoints[self.path])
        else:
            self.send_error(404)

    def handle_request(self, handler_func):
        try:
            content_length = int(self.headers.get("Content-Length", 0))

            # Handle empty requests
            if content_length == 0:
                data = {}
            else:
                post_data = self.rfile.read(content_length)
                data_str = post_data.decode("utf-8").strip()
                if data_str:
                    data = json.loads(data_str)
                else:
                    data = {}

            result = handler_func(self, data)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), HPOHandler)
    print("HPO Benchmark HTTP Server starting on port 8000...")
    print("Endpoints:")
    print("- POST /objective_function")
    print("- POST /sample_config")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

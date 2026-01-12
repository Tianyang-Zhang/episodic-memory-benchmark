import sys
from pathlib import Path

# Add parent directory to Python path so epbench imports work
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent.parent  # Go up from memmachine/benchmark_generate.py to episodic-memory-benchmark/
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse  # noqa: E402
import logging  # noqa: E402

from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("Starting quickstart.py script")
logger.info("=" * 80)

git_repo_filepath = '/Users/jinggong/memmachine/episodic-memory-benchmark'
data_folder = Path(git_repo_filepath) / 'epbench' / 'data'
env_file = Path(git_repo_filepath) / '.env'
logger.info("Default paths initialized:")
logger.info(f"  - Git repo path: {git_repo_filepath}")
logger.info(f"  - Data folder: {data_folder}")
logger.info(f"  - Env file: {env_file}")

# Parsing the arguments
logger.info("Parsing command-line arguments...")
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default=str(data_folder),
                    help='Path to the data folder')
parser.add_argument('--env_file', type=str, default=str(env_file),
                    help='Path to the .env file')
parser.add_argument('--book_nb_events', type=int, default=20,
                    help='Number of events in the book (20 for short and 200 for long, for the default experiment)')


# Overrid the file paths
args = parser.parse_args()
data_folder = Path(args.data_folder)
env_file = Path(args.env_file)
logger.info("Arguments parsed successfully:")
logger.info(f"  - data_folder: {data_folder}")
logger.info(f"  - env_file: {env_file}")
logger.info(f"  - book_nb_events: {args.book_nb_events}")

# Step 1: generating the synthetic episodic memory dataset
logger.info("")
logger.info("=" * 80)
logger.info("STEP 1: Generating the synthetic episodic memory dataset")
logger.info("=" * 80)

# Configuration (here, default short book with 20 events)
logger.info("Configuring parameters for benchmark generation...")
book_parameters = {
    'indexing': 'default',
    'nb_summaries': 0
}
prompt_parameters = {
    'nb_events': args.book_nb_events,
    'name_universe': 'default',
    'name_styles': 'default',
    'seed': 0,
    'distribution_events': {
        'name': 'geometric',
        'param': 0.1
    }
}
model_parameters = {
    'model_name': 'claude-3-5-sonnet-20240620',
    'max_new_tokens': 4096,
    'itermax': 10
}
logger.info("Configuration parameters:")
logger.info(f"  - book_parameters: {book_parameters}")
logger.info(f"  - prompt_parameters: {prompt_parameters}")
logger.info(f"  - model_parameters: {model_parameters}")

# Generation (generate the book, then compute the ground truth QAs)
logger.info("")
logger.info("Creating BenchmarkGenerationWrapper and generating benchmark...")
my_benchmark = BenchmarkGenerationWrapper(
    prompt_parameters, model_parameters, book_parameters, data_folder, env_file)
logger.info("✓ Benchmark generation completed")
logger.info(f"  - Benchmark object created: {type(my_benchmark).__name__}")
logger.info(f"  - Benchmark pretty_print_debug_event_idx: {my_benchmark.pretty_print_debug_event_idx()}")

